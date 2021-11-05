"""
--------------------------------------------------------------------------------
Test MAGIC-LAT over different regularization parameters.
--------------------------------------------------------------------------------

Description: Computes MAGIC-LAT error metrics for test sets of 100 vertices with
either coarse or fine regularization parameters.  Results written out to text
files.

Results independently plotted.

Requirements: os, argparse, numpy, math, random

File: params.py

Date: 11/05/2021

Author: Jennifer Hellar
Email: jenniferhellar@gmail.com
--------------------------------------------------------------------------------
"""

import os
import argparse

import numpy as np
import math
import random

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT

import utils
import metrics
from const import DATADIR, DATAFILES
from magicLAT import magicLAT


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


""" Parse the input for data index argument. """
parser = argparse.ArgumentParser(
    description='Processes a single mesh file repeatedly for comparison of MAGIC-LAT, GPR, and quLATi performance.')

parser.add_argument('-i', '--idx', required=True, default='11',
                    help='Data index to process. \
                    Default: 11')

parser.add_argument('-a', '--anomalies_removed', required=False, default=1,
                    help='Remove anomalous points (disable: 0, enable: 1). \
                    Default: 1')

parser.add_argument('-r', '--repeat', required=True, default=20,
                    help='Number of test repetitions. \
                    Default: 20')

args = parser.parse_args()

PATIENT_IDX				=		int(vars(args)['idx'])
NUM_TEST_REPEATS		=		int(vars(args)['repeat'])
remove_anomalies		=		int(vars(args)['anomalies_removed'])

""" Obtain file names, patient number, mesh id, etc. """
(meshFile, latFile, ablFile) = DATAFILES[PATIENT_IDX]

nm = meshFile[0:-5]
patient = nm[7:10]
id = latFile.split('_')[3]

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

resFileDE2000 = os.path.join(resDir, 'p{}_t{:g}_m{:g}_r{:g}_de2000.txt'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS, NUM_TEST_REPEATS))
with open(resFileDE2000, 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}'.format('alpha', 'beta', 'mean', 'std'))

print('Reading files for ' + nm + ' ...\n')
[vertices, faces] = readMesh(os.path.join(DATADIR, meshFile))
[OrigLatCoords, OrigLatVals] = readLAT(os.path.join(DATADIR, latFile))

n = len(vertices)

mapIdx = [i for i in range(n)]
mapCoord = [vertices[i] for i in mapIdx]

allLatIdx, allLatCoord, allLatVal = utils.mapSamps(mapIdx, mapCoord, OrigLatCoords, OrigLatVals)

M = len(allLatIdx)

# Identify and exclude anomalous LAT samples
anomalous = np.zeros(M)
if remove_anomalies:
	anomIdx = []	
	if PATIENT_IDX == 4:
		anomIdx = [25, 112, 159, 218, 240, 242, 264]
	elif PATIENT_IDX == 5:
		anomIdx = [119, 150, 166, 179, 188, 191, 209, 238]
	elif PATIENT_IDX == 6:
		anomIdx = [11, 12, 59, 63, 91, 120, 156]
	elif PATIENT_IDX ==7:
		anomIdx = [79, 98, 137, 205]
	elif PATIENT_IDX == 8:
		anomIdx = [10, 11, 51, 56, 85, 105, 125, 143, 156, 158, 169, 181, 210, 269, 284, 329, 336, 357, 365, 369, 400, 405]
	elif PATIENT_IDX == 9:
		anomIdx = [0, 48, 255, 322]
	else:
		anomalous = utils.isAnomalous(allLatCoord, allLatVal)
	anomalous[anomIdx] = 1
else:
	anomalous = [0 for i in range(M)]

numPtsIgnored = np.sum(anomalous)

latIdx = [allLatIdx[i] for i in range(M) if anomalous[i] == 0]
latCoords = [allLatCoord[i] for i in range(M) if anomalous[i] == 0]
latVals = [allLatVal[i] for i in range(M) if anomalous[i] == 0]

M = len(latIdx)

mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]



""" Sampling """
sampLst = utils.getModifiedSampList(latVals)

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
			latEst = magicLAT(vertices, faces, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD, alpha, beta)


			""" Error metrics """
			TstVal = [mapLAT[i] for i in TstIdx]
			TstValEst = latEst[TstIdx]

			magicNMSE[test] = metrics.calcNMSE(TstVal, TstValEst)
			magicSNR[test] = metrics.calcSNR(TstVal, TstValEst)
			magicMAE[test] = metrics.calcMAE(TstVal, TstValEst)
			magicDE2000[test] = metrics.deltaE(TstVal, TstValEst, MINLAT, MAXLAT)


		with open(resFileNMSE, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicNMSE), np.std(magicNMSE)))

		with open(resFileSNR, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicSNR), np.std(magicSNR)))

		with open(resFileMAE, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicMAE), np.std(magicMAE)))

		with open(resFileDE2000, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}'.format(alpha, beta, np.average(magicDE2000), np.std(magicDE2000)))


print('\nTest complete.\n')