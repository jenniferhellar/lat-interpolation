"""
--------------------------------------------------------------------------------
Manifold Approximating Graph Interpolation on Cardiac mapLAT data (MAGIC-mapLAT).
--------------------------------------------------------------------------------

Description: Cross-validation to randomly select test sets for interpolation.  
5x repetitition for error mean and variance estimation.

Requirements: os, numpy, matplotlib, sklearn, scipy, math

File: gpr_interp.py

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""

import os

import numpy as np
import math
import random

# plotting packages
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import cv2

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


from utils import *
from const import *
from magicLAT import *


# 20 is the one I have been working with
PATIENT_MAP				=		20

NUM_TEST_SAMPLES 		= 		100
NUM_TEST_REPEATS 		= 		5
EDGE_THRESHOLD			=		50

""" Read the files """
meshFile = meshNames[PATIENT_MAP]
latFile = latNames[PATIENT_MAP]
nm = meshFile[0:-5]
patient = nm[7:10]

print('Reading files for ' + nm + ' ...\n')
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[latCoords, latVals] = readLAT(dataDir + latFile)

n = len(coordinateMatrix)

mapIdx = [i for i in range(n)]
mapCoord = [coordinateMatrix[i] for i in mapIdx]

latIdx, latCoords, latVals = mapSamps(mapIdx, mapCoord, latCoords, latVals)

M = len(latIdx)
if NUM_TEST_SAMPLES > M:
	print('Not enough known samples for this experiment. NUM_TEST_SAMPLES must be <{:g}.\n'.format(M))
	exit(1)

mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]

sampLst = [i for i in range(M)]

testsNMSE = [0 for i in range(NUM_TEST_REPEATS)]
testsSNR = [0 for i in range(NUM_TEST_REPEATS)]
testsPerc = [0 for i in range(NUM_TEST_REPEATS)]
testsSSIM = [0 for i in range(NUM_TEST_REPEATS)]
testsMAE = [0 for i in range(NUM_TEST_REPEATS)]
testsNRMSE = [0 for i in range(NUM_TEST_REPEATS)]

if not os.path.isdir('results'):
	os.makedirs('results')


for test in range(NUM_TEST_REPEATS):
	
	print('test #{:g}'.format(test))

	tr_i = random.sample(sampLst, NUM_TEST_SAMPLES)
	tst_i = [i for i in range(M) if i not in tr_i]

	# number of labelled and unlabelled vertices in this fold
	trLen = len(tr_i)
	tstLen = len(tst_i)

	# get vertex indices of labelled/unlabelled nodes
	TrIdx = sorted(np.take(latIdx, tr_i))
	TstIdx = sorted(np.take(latIdx, tst_i))

	# get vertex coordinates
	TrCoord = [mapCoord[i] for i in TrIdx]
	TstCoord = [mapCoord[i] for i in TstIdx]

	# get mapLAT signal values
	TrVal = [mapLAT[i] for i in TrIdx]
	TstVal = [mapLAT[i] for i in TstIdx]

	latEstFold = magicLAT(coordinateMatrix, connectivityMatrix, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)

	nmse = calcNMSE(TstVal, latEstFold[TstIdx])
	snr = calcSNR(TstVal, latEstFold[TstIdx])
	perc = calcPercError(TstVal, latEstFold[TstIdx])
	mae = calcMAE(TstVal, latEstFold[TstIdx])
	nrmse = calcNRMSE(TstVal, latEstFold[TstIdx])

	testsNMSE[test] = nmse
	testsSNR[test] = snr
	testsPerc[test] = perc
	testsMAE[test] = mae
	testsNRMSE[test] = nrmse

	# For colorbar ranges
	MINLAT = math.floor(min(latVals)/10)*10
	MAXLAT = math.ceil(max(latVals)/10)*10

	# Figure view
	elev = 24
	azim = -135

	pltSig = TrVal
	pltCoord = np.array(TrCoord)
	triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

	fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8), subplot_kw=dict(projection="3d"))
	axes = ax.flatten()

	# Plot true mapLAT signal
	thisAx = axes[0]
	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	# thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey')
	pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

	thisAx.set_title('Input Samples\nm = {:g}'.format(trLen))
	thisAx.set_xlabel('X', fontweight ='bold') 
	thisAx.set_ylabel('Y', fontweight ='bold') 
	thisAx.set_zlabel('Z', fontweight ='bold')
	thisAx.view_init(elev, azim)
	cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01])
	plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

	thisAx = axes[1]

	pltSig = latEstFold
	pltCoord = coordinateMatrix

	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

	thisAx.set_title('Interpolation\nNMSE = {:.4f}, SNR = {:.4f}'.format(nmse, snr))
	thisAx.set_xlabel('X', fontweight ='bold') 
	thisAx.set_ylabel('Y', fontweight ='bold') 
	thisAx.set_zlabel('Z', fontweight ='bold')
	thisAx.view_init(elev, azim)
	# cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height]) # vertical bar to the right
	cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
	plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

	filename = os.path.join('results', 'p{}_t{:g}_m{:g}_test{:g}_entire.png'.format(patient, EDGE_THRESHOLD, NUM_TEST_SAMPLES, test))
	fig.savefig(filename)
	plt.close()

	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8), subplot_kw=dict(projection="3d"))
	# axes = ax.flatten()
	# thisAx = axes[0]
	thisAx = ax

	pltSig = TstVal
	pltCoord = np.array(TstCoord)

	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

	plt.axis('off')
	thisAx.grid(b=None)

	thisAx.set_title('Ground truth\n')
	thisAx.set_xlabel('X', fontweight ='bold') 
	thisAx.set_ylabel('Y', fontweight ='bold') 
	thisAx.set_zlabel('Z', fontweight ='bold')
	thisAx.view_init(elev, azim)
	cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
	plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

	truthFilename = os.path.join('results', 'p{}_t{:g}_m{:g}_test{:g}_truth.png'.format(patient, EDGE_THRESHOLD, NUM_TEST_SAMPLES, test))
	fig.savefig(truthFilename)
	plt.close()


	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8), subplot_kw=dict(projection="3d"))
	# axes = ax.flatten()
	# thisAx = axes[0]
	thisAx = ax

	pltSig = latEstFold[TstIdx]
	pltCoord = np.array(TstCoord)

	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

	plt.axis('off')
	thisAx.grid(b=None)

	thisAx.set_title('Estimate\nNMSE = {:.4f}, SNR = {:.4f}'.format(nmse, snr))
	thisAx.set_xlabel('X', fontweight ='bold') 
	thisAx.set_ylabel('Y', fontweight ='bold') 
	thisAx.set_zlabel('Z', fontweight ='bold')
	thisAx.view_init(elev, azim)
	# cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height]) # vertical bar to the right
	cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
	plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

	filename = os.path.join('results', 'p{}_t{:g}_m{:g}_test{:g}_testpts.png'.format(patient, EDGE_THRESHOLD, NUM_TEST_SAMPLES, test))
	fig.savefig(filename)
	plt.close()

	# figTruth = cv2.imread(truthFilename)
	# figEst = cv2.imread(filename)

	# testsSSIM[test] = calcSSIM(figTruth, figEst)

	# plt.show()

filename = os.path.join('results', 'p{}_t{:g}_m{:g}_tests{:g}_metrics.txt'.format(patient, EDGE_THRESHOLD, NUM_TEST_SAMPLES, NUM_TEST_REPEATS))
fid = open(filename, 'w')
fid.write('{:<30}{}\n'.format('file', nm))
fid.write('{:<30}{:g}\n'.format('n', n))
fid.write('{:<30}{:g}\n\n'.format('m', M))

fid.write('{:<30}{:g}\n'.format('NUM_TEST_SAMPLES', NUM_TEST_SAMPLES))
fid.write('{:<30}{:g}\n'.format('EDGE_THRESHOLD', EDGE_THRESHOLD))
fid.write('{:<30}{:g}\n\n'.format('NUM_TEST_REPEATS', NUM_TEST_REPEATS))

fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('NMSE', np.average(testsNMSE), np.std(testsNMSE)))
fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('SNR', np.average(testsSNR), np.std(testsSNR)))
fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('Perc', np.average(testsPerc), np.std(testsPerc)))
fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('SSIM', np.average(testsSSIM), np.std(testsSSIM)))
fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('MAE', np.average(testsMAE), np.std(testsMAE)))
fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('nRMSE', np.average(testsNRMSE), np.std(testsNRMSE)))
fid.close()