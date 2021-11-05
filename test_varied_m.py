
"""
--------------------------------------------------------------------------------
Executes GPR, GPMI, and MAGIC-LAT across multiple input sizes and records the 
performance metric results.
--------------------------------------------------------------------------------

usage: test_varied_m.py [-h] -i IDX [-a ANOMALIES_REMOVED] -r REPEAT

Tests MAGIC-LAT, GPR, and quLATi performance across multiple input sizes.

optional arguments:
  -h, --help            show this help message and exit
  -i IDX, --idx IDX     Data index to process. Default: 11
  -a ANOMALIES_REMOVED, --anomalies_removed ANOMALIES_REMOVED
                        Remove anomalous points (disable: 0, enable: 1). Default: 1
  -r REPEAT, --repeat REPEAT
                        Number of test repetitions. Default: 25

DATA INDICES:
	Too large for my laptop:
		p031 = 0 (4-SINUS LVFAM)
		p032 = 1 (1-LVFAM LAT HYB), 2 (2-LVFAM INITIAL PVC), 3 (4-LVFAM SINUS)
		p037 = 10 (12-LV-SINUS)

	Testable:
		p033 = 4 (3-RV-FAM-PVC-A-NORMAL), 5 (4-RV-FAM-PVC-A-LAT-HYBRID)
		p034 = 6 (4-RVFAM-LAT-HYBRID), 7 (5-RVFAM-PVC), 8 (6-RVFAM-SINUS-VOLTAGE)
		p035 = 9 (8-SINUS)
		p037 = 11 (9-RV-SINUS-VOLTAGE)

Requirements: 
	os, argparse,
	numpy, math, random, 
	vedo, scikit-learn,
	quLATi, robust_laplacian

Author: Jennifer Hellar
Email: jenniferhellar@gmail.com
"""

import os

import argparse

import numpy as np
import math
import random

# Gaussian process regression interpolation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


import utils
import metrics
from const import DATADIR, DATAFILES
from magicLAT import magicLAT

import quLATiHelper


EDGE_THRESHOLD			=		50

OUTDIR				 	=		'test_varied_m_results'

m						=		[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

""" Parse the input for data index argument. """
parser = argparse.ArgumentParser(
    description='Tests MAGIC-LAT, GPR, and quLATi performance across multiple input sizes.')

parser.add_argument('-i', '--idx', required=True, default='11',
                    help='Data index to process. \
                    Default: 11')

parser.add_argument('-a', '--anomalies_removed', required=False, default=1,
                    help='Remove anomalous points (disable: 0, enable: 1). \
                    Default: 1')

parser.add_argument('-r', '--repeat', required=True, default=25,
                    help='Number of test repetitions. \
                    Default: 25')

args = parser.parse_args()

PATIENT_IDX				=		int(vars(args)['idx'])
NUM_TEST_REPEATS		=		int(vars(args)['repeat'])
remove_anomalies		=		int(vars(args)['anomalies_removed'])

""" Obtain file names, patient number, mesh id, etc. """
(meshFile, latFile, ablFile) = DATAFILES[PATIENT_IDX]

nm = meshFile[0:-5]
patient = nm[7:10]
id = latFile.split('_')[3]

""" Create output directory for this script and subdir for this mesh. """
if not os.path.isdir(OUTDIR):
	os.makedirs(OUTDIR)

""" Read the files """
print('\nProcessing ' + nm + ' ...\n')
[vertices, faces] = readMesh(os.path.join(DATADIR, meshFile))
[OrigLatCoords, OrigLatVals] = readLAT(os.path.join(DATADIR, latFile))

if ablFile != '':
	ablFile = os.path.join(DATADIR, ablFile)
else:
	ablFile = None
	print('No ablation file available for this mesh... continuing...\n')

""" Pre-process the mesh and LAT samples. """
n = len(vertices)

mapIdx = [i for i in range(n)]
mapCoord = [vertices[i] for i in mapIdx]

# Map the LAT samples to nearest mesh vertices
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

# For colorbar ranges
MINLAT = min(latVals)
MAXLAT = max(latVals)

# Create partially-sampled signal vector
mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]

""" Create GPR kernel and regressor """
gp_kernel = RBF(length_scale=0.01) + RBF(length_scale=0.1) + RBF(length_scale=1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)

"""
Create the quLATi model
"""
model = quLATiHelper.quLATiModel(patient, vertices, faces)

infoFile = os.path.join(OUTDIR, 'p' + patient + '_' + id.split('-')[0] + '_info.txt')
meanFile = os.path.join(OUTDIR, 'p' + patient + '_' + id.split('-')[0] + '_mean.txt')
stdFile = os.path.join(OUTDIR, 'p' + patient + '_' + id.split('-')[0] + '_std.txt')

with open(infoFile, 'w') as fid:
	fid.write('{:<30}{}\n'.format('file', nm))
	fid.write('{:<30}{:g}\n'.format('n', n))
	fid.write('{:<30}{:g}\n'.format('M', M))
	fid.write('{:<30}{:g}\n'.format('ignored', np.sum(anomalous)))

	fid.write('{:<30}{:g}\n'.format('EDGE_THRESHOLD', EDGE_THRESHOLD))
	fid.write('{:<30}{:g}\n'.format('NUM_TEST_REPEATS', NUM_TEST_REPEATS))

with open(meanFile, 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}'.format('m', 'MAGIC-LAT', 'GPR', 'quLATi'))

with open(stdFile, 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}'.format('m', 'MAGIC-LAT', 'GPR', 'quLATi'))

""" Random train/test split by non-uniform sampling distribution. """
# list with values repeated proportionally to sampling probability
sampLst = utils.getModifiedSampList(latVals)

for i in range(len(m)):

	numSamps = m[i]

	print('testing #{:g} of {:g} possible m values.'.format(i + 1, len(m)))

	magicDE = [0 for i in range(NUM_TEST_REPEATS)]
	gprDE = [0 for i in range(NUM_TEST_REPEATS)]
	quLATiDE = [0 for i in range(NUM_TEST_REPEATS)]

	for test in range(NUM_TEST_REPEATS):
		
		print('\ttest #{:g} of {:g}.'.format(test + 1, NUM_TEST_REPEATS))

		samps = sampLst

		tr_i = []
		for i in range(numSamps):
			elem = random.sample(samps, 1)[0]
			tr_i.append(elem)
			samps = [i for i in samps if i != elem]	# to prevent repeats
		tst_i = [i for i in range(M) if i not in tr_i]

		# get vertex indices of labelled/unlabelled nodes
		TrIdx = sorted(np.take(latIdx, tr_i))
		TstIdx = sorted(np.take(latIdx, tst_i))

		# get vertex coordinates
		TrCoord = [vertices[i] for i in TrIdx]
		TstCoord = [vertices[i] for i in TstIdx]

		# get mapLAT signal values
		TrVal = [mapLAT[i] for i in TrIdx]
		TstVal = [mapLAT[i] for i in TstIdx]


		""" MAGIC-LAT estimate """
		latEst = magicLAT(vertices, faces, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)

		""" GPR estimate """
		gpr.fit(TrCoord, TrVal)
		latEstGPR = gpr.predict(vertices, return_std=False)

		""" quLATi estimate """
		latEstquLATi = quLATiHelper.quLATi(TrIdx, TrVal, vertices, model)


		magicDE[test] = metrics.deltaE(TstVal, latEst[TstIdx], MINLAT, MAXLAT)
		gprDE[test] = metrics.deltaE(TstVal, latEstGPR[TstIdx], MINLAT, MAXLAT)
		quLATiDE[test] = metrics.deltaE(TstVal, latEstquLATi[TstIdx], MINLAT, MAXLAT)

	with open(meanFile, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}'.format(numSamps, np.average(magicDE), np.average(gprDE), np.average(quLATiDE)))

	with open(stdFile, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}'.format(numSamps, np.std(magicDE), np.std(gprDE), np.std(quLATiDE)))