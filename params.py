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
p037 = 21
"""
PATIENT_MAP				=		14

NUM_TEST_REPEATS 		= 		50
NUM_TRAIN_SAMPS 		= 		100
EDGE_THRESHOLD			=		50

COARSE 					=		1

""" Hyperparameters """
if COARSE:
	alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
	betas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
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

resFileDE1976 = os.path.join(resDir, 'p{}_t{:g}_m{:g}_r{:g}_de1976.txt'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS, NUM_TEST_REPEATS))
with open(resFileDE1976, 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}'.format('alpha', 'beta', 'mean', 'std'))

resFileDE2000 = os.path.join(resDir, 'p{}_t{:g}_m{:g}_r{:g}_de2000.txt'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS, NUM_TEST_REPEATS))
with open(resFileDE2000, 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}'.format('alpha', 'beta', 'mean', 'std'))

print('Reading files for ' + nm + ' ...\n')
[vertices, faces] = readMesh(os.path.join(dataDir, meshFile))
[OrigLatCoords, OrigLatVals] = readLAT(os.path.join(dataDir, latFile))

n = len(vertices)

mapIdx = [i for i in range(n)]
mapCoord = [vertices[i] for i in mapIdx]

allLatIdx, allLatCoord, allLatVal = mapSamps(mapIdx, mapCoord, OrigLatCoords, OrigLatVals)

M = len(allLatIdx)

anomalous = isAnomalous(allLatCoord, allLatVal, k=6, d=5, thresh=50)

numPtsIgnored = np.sum(anomalous)

latIdx = [allLatIdx[i] for i in range(M) if anomalous[i] == 0]
latCoords = [allLatCoord[i] for i in range(M) if anomalous[i] == 0]
latVals = [allLatVal[i] for i in range(M) if anomalous[i] == 0]

M = len(latIdx)

mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]

edgeFile = os.path.join('..', 'E_p{}.npy'.format(patient))
if not os.path.isfile(edgeFile):
	[edges, TRI] = edgeMatrix(vertices, faces)

	print('Writing edge matrix to file...')
	with open(edgeFile, 'wb') as fid:
		np.save(fid, edges)
else:
	edges = np.load(edgeFile, allow_pickle=True)



""" Sampling """
sampLst = getModifiedSampList(latVals)

MINLAT = math.floor(min(allLatVal)/10)*10
MAXLAT = math.ceil(max(allLatVal)/10)*10

for a_idx in range(len(alphas)):
	alpha = alphas[a_idx]

	for b_idx in range(len(betas)):
		beta = betas[b_idx]

		print('\nCross-validating alpha=' + str(alpha) + ', beta=' + str(beta))

		magicNMSE = [0 for i in range(NUM_TEST_REPEATS)]
		magicSNR = [0 for i in range(NUM_TEST_REPEATS)]
		magicMAE = [0 for i in range(NUM_TEST_REPEATS)]
		magicDE1976 = [0 for i in range(NUM_TEST_REPEATS)]
		magicDE2000 = [0 for i in range(NUM_TEST_REPEATS)]

		for test in range(NUM_TEST_REPEATS):
			
			print('test #{:g} of {:g}.'.format(test + 1, NUM_TEST_REPEATS))

			samps = sampLst

			tr_i = []
			for i in range(NUM_TRAIN_SAMPS):
				elem = random.sample(samps, 1)[0]
				tr_i.append(elem)
				samps = [i for i in samps if i != elem]	# to prevent repeats
			tst_i = [i for i in range(M) if i not in tr_i]

			# get map indices of training/test vertices
			TrIdx = sorted(np.take(latIdx, tr_i))
			TstIdx = sorted(np.take(latIdx, tst_i))

			# get training values and coordinates
			TrVal = [mapLAT[i] for i in TrIdx]
			TrCoord = [mapCoord[i] for i in TrIdx]


			""" MAGIC-LAT estimate """
			latEst = magicLAT(vertices, faces, edges, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD, alpha, beta)


			""" Error metrics """
			TstVal = [mapLAT[i] for i in TstIdx]
			TstValEst = latEst[TstIdx]

			magicNMSE[test] = calcNMSE(TstVal, TstValEst)
			magicSNR[test] = calcSNR(TstVal, TstValEst)
			magicMAE[test] = calcMAE(TstVal, TstValEst)
			magicDE1976[test], magicDE2000[test] = deltaE(TstVal, TstValEst, MINLAT, MAXLAT)


		with open(resFileNMSE, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicNMSE), np.std(magicNMSE)))

		with open(resFileSNR, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicSNR), np.std(magicSNR)))

		with open(resFileMAE, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicMAE), np.std(magicMAE)))

		with open(resFileDE1976, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicDE1976), np.std(magicDE1976)))

		with open(resFileDE2000, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicDE2000), np.std(magicDE2000)))


print('\nTest complete.\n')