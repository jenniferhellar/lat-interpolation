"""
--------------------------------------------------------------------------------
Test MAGIC-LAT over different regularization parameters.
--------------------------------------------------------------------------------

Description: Computes MAGIC-LAT error metrics for test sets of 100 vertices with
either coarse or fine regularization parameters.  Results written out to text
files.

Results independently plotted in 
	reg_params_coarse.png
	reg_params_fine.png.

Requirements: os, numpy, sklearn, scipy, math

File: test_reg_params.py

Date: 05/24/2021

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""

import time

import os

import numpy as np
import math
import random

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
PATIENT_MAP				=		9

NUM_TEST_REPEATS 		= 		20
NUM_TRAIN_SAMPS 		= 		100
EDGE_THRESHOLD			=		50

COARSE 					=		1

""" Hyperparameters """
if COARSE:
	alphas = [0.01, 0.1, 1, 5, 10, 100, 1000]
	betas = [0.01, 0.1, 1, 5, 10, 100, 1000]
else:
	print('fine-tuned params not selected')
	exit(0)

""" Read the files """
meshFile = meshNames[PATIENT_MAP]
latFile = latNames[PATIENT_MAP]
nm = meshFile[0:-5]
patient = nm[7:10]

# create a results directory
resDir = os.path.join('..','res_reg')
if not os.path.isdir(resDir):
	os.makedirs(resDir)

resFileNMSE = os.path.join(resDir, 'p{}_t{:g}_m{:g}_r{:g}_nmse.txt'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS, NUM_TEST_REPEATS))
with open(resFileNMSE, 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}'.format('alpha', 'beta', 'mean', 'std'))

resFileSNR = os.path.join(resDir, 'p{}_t{:g}_m{:g}_r{:g}_snr.txt'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS, NUM_TEST_REPEATS))
with open(resFileSNR, 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}'.format('alpha', 'beta', 'mean', 'std'))

resFileMAE = os.path.join(resDir, 'p{}_t{:g}_m{:g}_r{:g}_mae.txt'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS, NUM_TEST_REPEATS))
with open(resFileMAE, 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}'.format('alpha', 'beta', 'mean', 'std'))

resFileNRMSE = os.path.join(resDir, 'p{}_t{:g}_m{:g}_r{:g}_nrmse.txt'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS, NUM_TEST_REPEATS))
with open(resFileNRMSE, 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}'.format('alpha', 'beta', 'mean', 'std'))

print('Reading files for ' + nm + ' ...\n')
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[allLatCoord, allLatVal] = readLAT(dataDir + latFile)

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

	neighbors = [nearestVers[i, n] for n in range(1,k) if dist[i,n] < 10]

	cnt = 0
	for neighVer in neighbors:
		neighVal = allLatVal[neighVer]

		if abs(verVal - neighVal) > 50:
			cnt += 1
		else:
			break

	if cnt == len(neighbors):
		anomalous[i] = 1

latIdx = [allLatIdx[i] for i in range(M) if anomalous[i] == 0]
latCoords = [allLatCoord[i] for i in range(M) if anomalous[i] == 0]
latVals = [allLatVal[i] for i in range(M) if anomalous[i] == 0]

M = len(latIdx)

mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]

edgeFile = os.path.join('..', 'E_p{}.npy'.format(patient))
if not os.path.isfile(edgeFile):
	[EDGES, TRI] = edgeMatrix(coordinateMatrix, connectivityMatrix)

	print('Writing edge matrix to file...')
	with open(edgeFile, 'wb') as fid:
		np.save(fid, EDGES)
else:
	EDGES = np.load(edgeFile, allow_pickle=True)

sampLst = [i for i in range(M)]

for a_idx in range(len(alphas)):
	alpha = alphas[a_idx]

	for b_idx in range(len(betas)):
		beta = betas[b_idx]

		print('\nCross-validating alpha=' + str(alpha) + ', beta=' + str(beta))

		magicNMSE = [0 for i in range(NUM_TEST_REPEATS)]
		magicSNR = [0 for i in range(NUM_TEST_REPEATS)]
		magicMAE = [0 for i in range(NUM_TEST_REPEATS)]
		magicNRMSE = [0 for i in range(NUM_TEST_REPEATS)]

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
			latEst = magicLAT(coordinateMatrix, connectivityMatrix, EDGES, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD, alpha, beta)


			""" Error metrics """
			TstVal = [mapLAT[i] for i in TstIdx]
			TstValEst = latEst[TstIdx]

			nmse, snr, mae, nrmse = compute_metrics(TstVal, TstValEst)

			magicNMSE[test] = nmse
			magicSNR[test] = snr
			magicMAE[test] = mae
			magicNRMSE[test] = nrmse

		with open(resFileNMSE, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicNMSE), np.std(magicNMSE)))

		with open(resFileSNR, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicSNR), np.std(magicSNR)))

		with open(resFileMAE, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicMAE), np.std(magicMAE)))

		with open(resFileNRMSE, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicNRMSE), np.std(magicNRMSE)))


print('\nTest complete.\n')