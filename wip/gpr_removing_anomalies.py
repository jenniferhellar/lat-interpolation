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

# 9, 12, 20 is the one I have been working with
PATIENT_MAP				=		14

NUM_TRAIN_SAMPS 		= 		300
TEST_CHUNK_SIZE			=		10
NUM_TEST_REPEATS 		= 		2
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

# KD Tree to find the nearest mesh vertex
k = 6
coordKDtree = cKDTree(latCoords)
[dist, nearestVers] = coordKDtree.query(latCoords, k=k)

anomalous = np.zeros(M)

for i in range(M):
	verCoord = latCoords[i]
	verVal = latVals[i]

	neighbors = [nearestVers[i, n] for n in range(1,k) if dist[i,n] < 10]

	cnt = 0
	for neighVer in neighbors:
		neighVal = latVals[neighVer]

		if abs(verVal - neighVal) > 50:
			cnt += 1
		else:
			break

	if cnt == len(neighbors):
		anomalous[i] = 1

print(np.sum(anomalous))

# For colorbar ranges
MINLAT = math.floor(min(latVals)/10)*10
MAXLAT = math.ceil(max(latVals)/10)*10

# Figure view
elev = 24
azim = -135

triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)


# pltSig = [latVals[i] for i in range(M) if anomalous[i] == 1]
# pltCoord = [latCoords[i] for i in range(M) if anomalous[i] == 1]
# pltCoord = np.array(pltCoord)

# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8), subplot_kw=dict(projection="3d"))
# thisAx = ax

# thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
# pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

# thisAx.set_title('Removed points')
# thisAx.set_xlabel('X', fontweight ='bold') 
# thisAx.set_ylabel('Y', fontweight ='bold') 
# thisAx.set_zlabel('Z', fontweight ='bold')
# thisAx.view_init(elev, azim)
# cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
# plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

# plt.show()

allLatIdx = latIdx
allLatCoord = latCoords
allLatVal = latVals

latIdx = [allLatIdx[i] for i in range(M) if anomalous[i] == 0]
latCoords = [allLatCoord[i] for i in range(M) if anomalous[i] == 0]
latVals = [allLatVal[i] for i in range(M) if anomalous[i] == 0]

M = len(latIdx)

fid = open(os.path.join('results_gpr','gpr_p{}_t{:g}_m{:g}_latVals.txt'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS)), 'w')
np.array(latVals).tofile(fid, sep='\n', format='%.5f')
fid.close()

pltSig = latVals
pltCoord = np.array(latCoords)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8), subplot_kw=dict(projection="3d"))
thisAx = ax

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

thisAx.set_title('Kept points')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

# plt.show()



M = len(latIdx)
if NUM_TRAIN_SAMPS > M:
	print('Not enough known samples for this experiment. NUM_TRAIN_SAMPS must be <{:g}.\n'.format(M))
	exit(1)

mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]




if not os.path.isdir('results_gpr'):
	os.makedirs('results_gpr')

""" Create GPR kernel and regressor """
gp_kernel = RBF(length_scale=0.01) + RBF(length_scale=0.1) + RBF(length_scale=1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)


resLAT = np.zeros(M)
sampLst = [i for i in range(M)]
random.shuffle(sampLst)
startIdx = 0
endIdx = startIdx + TEST_CHUNK_SIZE

iter = 0

while startIdx < M:
	# start = time.time()

	print('iteration #{:g} of {:g}.'.format(iter+1, math.ceil(M/TEST_CHUNK_SIZE)))
	iter += 1

	tst_i = sampLst[startIdx : endIdx]
	availableTr = [i for i in sampLst if i not in tst_i]

	tr_i = random.sample(availableTr, NUM_TRAIN_SAMPS)

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

	# fit the GPR with training samples
	gpr.fit(TrCoord, TrVal)

	# predict the entire signal
	latEstFold = gpr.predict(mapCoord, return_std=False)

	for idx in tst_i:
		verIdx = latIdx[idx]
		resLAT[idx] = latEstFold[verIdx]

	startIdx += TEST_CHUNK_SIZE
	endIdx = min(M, startIdx + TEST_CHUNK_SIZE)


nmse = calcNMSE(latVals, resLAT)

fid = open(os.path.join('results_gpr','gpr_p{}_t{:g}_m{:g}_resLat.txt'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS)), 'w')
resLAT.tofile(fid, sep='\n', format='%.5f')
fid.close()

# For colorbar ranges
MINLAT = math.floor(min(latVals)/10)*10
MAXLAT = math.ceil(max(latVals)/10)*10

# Figure view
elev = 24
azim = -135

triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8), subplot_kw=dict(projection="3d"))
thisAx = ax

pltSig = latVals
pltCoord = np.array(latCoords)

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

thisAx.set_title('Ground truth\n')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

truthFilename = os.path.join('results_gpr', 'gpr_p{}_t{:g}_m{:g}_truth.png'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS))
fig.savefig(truthFilename)
plt.close()


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8), subplot_kw=dict(projection="3d"))
thisAx = ax

pltSig = resLAT
pltCoord = np.array(latCoords)

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

thisAx.set_title('Estimate\nNMSE = {:.4f}'.format(nmse))
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

filename = os.path.join('results_gpr', 'gpr_p{}_t{:g}_m{:g}_testpts.png'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS))
fig.savefig(filename)
plt.close()