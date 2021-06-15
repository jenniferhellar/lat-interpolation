"""
--------------------------------------------------------------------------------
Manifold Approximating Graph Interpolation on Cardiac mapLAT data (MAGIC-LAT).
--------------------------------------------------------------------------------

Description: Cross-validation to randomly select test sets for interpolation.  
5x repetitition for error mean and variance estimation.

Requirements: os, numpy, matplotlib, sklearn, scipy, math

File: gpr_interp.py

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""

import time

import os

import numpy as np
import math
import random

# plotting packages
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import cv2

# Gaussian process regression interpolation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


from utils import *
from const import *
from magicLAT import *

"""
p033 = 9
p034 = 14
p035 = 18
p037 = 20
"""
PATIENT_MAP				=		14

NUM_TEST_REPEATS 		= 		20
NUM_TRAIN_SAMPS 		= 		300
EDGE_THRESHOLD			=		50

""" Read the files """
meshFile = meshNames[PATIENT_MAP]
latFile = latNames[PATIENT_MAP]
nm = meshFile[0:-5]
patient = nm[7:10]

print('Reading files for ' + nm + ' ...\n')
[coordinateMatrix, connectivityMatrix] = readMesh(os.path.join(dataDir, meshFile))
[allLatCoord, allLatVal] = readLAT(os.path.join(dataDir, latFile))

n = len(coordinateMatrix)

mapIdx = [i for i in range(n)]
mapCoord = [coordinateMatrix[i] for i in mapIdx]

allLatIdx, allLatCoord, allLatVal = mapSamps(mapIdx, mapCoord, allLatCoord, allLatVal)

M = len(allLatIdx)

# KD Tree to find the nearest mesh vertex
k = 6
coordKDtree = cKDTree(allLatCoord)
[dist, nearestVers] = coordKDtree.query(allLatCoord, k=k)

anomalous = np.zeros(M)

for i in range(M):
	verCoord = allLatCoord[i]
	verVal = allLatVal[i]

	neighbors = [nearestVers[i, n] for n in range(1,k) if dist[i,n] < 5]

	cnt = 0
	for neighVer in neighbors:
		neighVal = allLatVal[neighVer]

		if abs(verVal - neighVal) > 50:
			cnt += 1
		else:
			break

	if cnt == len(neighbors):
		anomalous[i] = 1

numPtsIgnored = np.sum(anomalous)

latIdx = [allLatIdx[i] for i in range(M) if anomalous[i] == 0]
latCoords = [allLatCoord[i] for i in range(M) if anomalous[i] == 0]
latVals = [allLatVal[i] for i in range(M) if anomalous[i] == 0]

M = len(latIdx)

print('{:<20}{:g}'.format('n', n))
print('{:<20}{:g}/{:g}'.format('m', NUM_TRAIN_SAMPS, M))
print('{:<20}{:g}'.format('ignored', numPtsIgnored))
print('\n')
# exit()

mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]

edgeFile = 'E_p{}.npy'.format(patient)
if not os.path.isfile(edgeFile):
	[EDGES, TRI] = edgeMatrix(coordinateMatrix, connectivityMatrix)

	print('Writing edge matrix to file...')
	with open(edgeFile, 'wb') as fid:
		np.save(fid, EDGES)
else:
	EDGES = np.load(edgeFile, allow_pickle=True)


""" Create GPR kernel and regressor """
gp_kernel = RBF(length_scale=0.01) + RBF(length_scale=0.1) + RBF(length_scale=1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)


if not os.path.isdir('results_repeated_comparison'):
	os.makedirs('results_repeated_comparison')

sampLst = [i for i in range(M)]


magicNMSE = [0 for i in range(NUM_TEST_REPEATS)]
magicSNR = [0 for i in range(NUM_TEST_REPEATS)]
magicMAE = [0 for i in range(NUM_TEST_REPEATS)]
magicNRMSE = [0 for i in range(NUM_TEST_REPEATS)]

gprNMSE = [0 for i in range(NUM_TEST_REPEATS)]
gprSNR = [0 for i in range(NUM_TEST_REPEATS)]
gprMAE = [0 for i in range(NUM_TEST_REPEATS)]
gprNRMSE = [0 for i in range(NUM_TEST_REPEATS)]

for test in range(NUM_TEST_REPEATS):
	
	print('test #{:g} of {:g}.'.format(test + 1, NUM_TEST_REPEATS))

	tr_i = random.sample(sampLst, NUM_TRAIN_SAMPS)
	tst_i = [i for i in sampLst if i not in tr_i]

	# get map indices of training/test vertices
	TrIdx = sorted(np.take(latIdx, tr_i))
	TstIdx = sorted(np.take(latIdx, tst_i))

	# get training values and coordinates
	TrVal = [mapLAT[i] for i in TrIdx]
	TrCoord = [mapCoord[i] for i in TrIdx]


	""" MAGIC-LAT estimate """
	latEst = magicLAT(coordinateMatrix, connectivityMatrix, EDGES, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)


	""" GPR estimate """
	gpr.fit(TrCoord, TrVal)
	latEstGPR = gpr.predict(mapCoord, return_std=False)


	""" Error metrics """
	TstVal = [mapLAT[i] for i in TstIdx]
	TstValEst = latEst[TstIdx]
	TstValEstGPR = latEstGPR[TstIdx]

	nmse, snr, mae, nrmse = compute_metrics(TstVal, TstValEst)

	magicNMSE[test] = nmse
	magicSNR[test] = snr
	magicMAE[test] = mae
	magicNRMSE[test] = nrmse

	nmse, snr, mae, nrmse = compute_metrics(TstVal, TstValEstGPR)

	gprNMSE[test] = nmse
	gprSNR[test] = snr
	gprMAE[test] = mae
	gprNRMSE[test] = nrmse

filename = os.path.join('results_repeated_comparison', 'p{}_t{:g}_m{:g}_tests{:g}.txt'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS, NUM_TEST_REPEATS))
with open(filename, 'w') as fid:
	fid.write('{:<30}{}\n'.format('file', nm))
	fid.write('{:<30}{:g}\n'.format('n', n))
	fid.write('{:<30}{:g}\n'.format('ignored', numPtsIgnored))
	fid.write('{:<30}{:g}/{:g}\n\n'.format('m', NUM_TRAIN_SAMPS, M))

	fid.write('{:<30}{:g}\n'.format('EDGE_THRESHOLD', EDGE_THRESHOLD))
	fid.write('{:<30}{:g}\n'.format('NUM_TEST_REPEATS', NUM_TEST_REPEATS))

	fid.write('\n\n')

	fid.write('MAGIC-LAT Performance\n\n')
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('NMSE', 	np.average(magicNMSE), 	np.std(magicNMSE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('SNR', 	np.average(magicSNR), 	np.std(magicSNR)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('MAE', 	np.average(magicMAE), 	np.std(magicMAE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('nRMSE', 	np.average(magicNRMSE), np.std(magicNRMSE)))

	fid.write('\n\n')

	fid.write('GPR Performance\n\n')
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('NMSE', 	np.average(gprNMSE), 	np.std(gprNMSE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('SNR', 	np.average(gprSNR), 	np.std(gprSNR)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('MAE', 	np.average(gprMAE), 	np.std(gprMAE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('nRMSE', 	np.average(gprNRMSE), 	np.std(gprNRMSE)))
