
"""

usage: test.py [-h] -i IDX [-v VERBOSE] [-t TEXT]

Processes a single mesh file for comparison of MAGIC-LAT, GPR, and quLATi performance.

optional arguments:
  -h, --help            show this help message and exit
  -i IDX, --idx IDX     Data index to process. Default: 11
  -v VERBOSE, --verbose VERBOSE
                        Verbose output (disable: 0, enable: 1). Default: 1
  -t TEXT, --text TEXT  Generate text-only outputs (disable: 0, enable: 1). Default: 0

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
	os, platform, argparse,
	numpy, math, random, 
	vedo, matplotlib, scikit-learn, scipy, cv2, colour
	quLATi, robust_laplacian
"""
from timeit import default_timer as timer

import os

import argparse

import numpy as np
import math
import random

# plotting packages
from vedo import Mesh

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



NUM_TRAIN_SAMPS 		= 		100
EDGE_THRESHOLD			=		50

OUTDIR				 	=		'test_results'


""" Parse the input for data index argument. """
parser = argparse.ArgumentParser(
    description='Processes a single mesh file for comparison of MAGIC-LAT, GPR, and quLATi performance.')

parser.add_argument('-i', '--idx', required=True, default='11',
                    help='Data index to process. \
                    Default: 11')

parser.add_argument('-a', '--anomalies_removed', required=True, default=1,
                    help='Remove anomalous points (disable: 0, enable: 1). \
                    Default: 1')

parser.add_argument('-v', '--verbose', required=False, default=1,
                    help='Verbose output (disable: 0, enable: 1). \
                    Default: 1')

parser.add_argument('-t', '--text', required=False, default=0,
                    help='Generate text-only outputs (disable: 0, enable: 1). \
                    Default: 0')

args = parser.parse_args()

PATIENT_IDX				=		int(vars(args)['idx'])
verbose					=		int(vars(args)['verbose'])
visualSuppressed		=		int(vars(args)['text'])
remove_anomalies		=		int(vars(args)['anomalies_removed'])

""" Obtain file names, patient number, mesh id, etc. """
(meshFile, latFile, ablFile) = DATAFILES[PATIENT_IDX]

nm = meshFile[0:-5]
patient = nm[7:10]
id = latFile.split('_')[3]

""" Create output directory for this script and subdir for this mesh. """
outSubDir = os.path.join(OUTDIR, 'p' + patient + '_' + id)
if not os.path.isdir(OUTDIR):
	os.makedirs(OUTDIR)
if not os.path.isdir(outSubDir):
	os.makedirs(outSubDir)

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
mesh = Mesh([vertices, faces])
mesh.c('grey')

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
MINLAT = math.floor(min(latVals)/10)*10
MAXLAT = math.ceil(max(latVals)/10)*10
if PATIENT_IDX == 4 or PATIENT_IDX == 5 or PATIENT_IDX == 8 or PATIENT_IDX == 9:
	MAXLAT = MINLAT + math.ceil((3/4 * (max(latVals) - MINLAT)) / 10)*10
elif PATIENT_IDX == 6 or PATIENT_IDX == 7 or PATIENT_IDX == 11:
	MAXLAT = MINLAT + math.ceil((7/8 * (max(latVals) - MINLAT)) / 10)*10

# Create partially-sampled signal vector
mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]


""" Random train/test split by non-uniform sampling distribution. """
# list with values repeated proportionally to sampling probability
sampLst = utils.getModifiedSampList(latVals)

tr_i = []
for i in range(NUM_TRAIN_SAMPS):
	elem = random.sample(sampLst, 1)[0]
	tr_i.append(elem)
	sampLst = [i for i in sampLst if i != elem]	# to prevent repeats
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
if verbose:
	print('\tBeginning MAGIC-LAT computation...')

# start = timer()
latEst = magicLAT(vertices, faces, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)
# stop = timer()
# print(stop-start)
# exit(0)


""" GPR estimate """
if verbose:
	print('\tBeginning GPR computation...')

gp_kernel = RBF(length_scale=0.01) + RBF(length_scale=0.1) + RBF(length_scale=1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)

# fit the GPR with training samples
gpr.fit(TrCoord, TrVal)

# predict the entire signal
latEstGPR = gpr.predict(vertices, return_std=False)


""" quLATi estimate """
if verbose:
	print('\tBeginning quLATi computation...')

model = quLATiHelper.quLATiModel(patient, vertices, faces)
latEstquLATi = quLATiHelper.quLATi(TrIdx, TrVal, vertices, model)


if not visualSuppressed:
	elev, azimuth, roll = utils.getPerspective(patient)

	"""
	Figure 0: Ground truth (entire), training points, and MAGIC-LAT (entire)
	"""
	utils.plotSaveEntire(mesh, latCoords, latVals, TrCoord, TrVal, latEst, 
		azimuth, elev, roll, MINLAT, MAXLAT,
		outSubDir, title='MAGIC-LAT', filename='magic', ablFile=ablFile)

	"""
	Figure 1: Ground truth (entire), training points, and GPR (entire)
	"""
	utils.plotSaveEntire(mesh, latCoords, latVals, TrCoord, TrVal, latEstGPR, 
		azimuth, elev, roll, MINLAT, MAXLAT,
		outSubDir, title='GPR', filename='gpr', ablFile=ablFile)

	"""
	Figure 2: Ground truth (entire), training points, and quLATi (entire)
	"""
	utils.plotSaveEntire(mesh, latCoords, latVals, TrCoord, TrVal, latEstquLATi, 
		azimuth, elev, roll, MINLAT, MAXLAT,
		outSubDir, title='quLATi', filename='quLATi', ablFile=ablFile)

	utils.plotSaveTwoColorMaps(mesh, latEst,
		azimuth, elev, roll, MINLAT, MAXLAT, outSubDir, 'gist_rainbow', 'viridis_r', filename='raw')

	utils.plotSaveIndividual(mesh, latCoords, latVals, TrCoord, TrVal, latEst, latEstGPR, latEstquLATi,
		azimuth, elev, roll, MINLAT, MAXLAT, outSubDir, idx=7, ablFile=ablFile)
"""
Error metrics
"""
if verbose:
	print('\tComputing metrics...')

nmse = metrics.calcNMSE(TstVal, latEst[TstIdx])
nmseGPR = metrics.calcNMSE(TstVal, latEstGPR[TstIdx])
nmsequLATi = metrics.calcNMSE(TstVal, latEstquLATi[TstIdx])

mae = metrics.calcMAE(TstVal, latEst[TstIdx])
maeGPR = metrics.calcMAE(TstVal, latEstGPR[TstIdx])
maequLATi = metrics.calcMAE(TstVal, latEstquLATi[TstIdx])

dE = metrics.deltaE(TstVal, latEst[TstIdx], MINLAT, MAXLAT)
dEGPR = metrics.deltaE(TstVal, latEstGPR[TstIdx], MINLAT, MAXLAT)
dEquLATi = metrics.deltaE(TstVal, latEstquLATi[TstIdx], MINLAT, MAXLAT)

if verbose:
	print('\tWriting to file...')

with open(os.path.join(outSubDir, 'metrics.txt'), 'w') as fid:
	fid.write(nm + '\n\n')
	fid.write('{:<20}{:g}\n'.format('n', n))
	fid.write('{:<20}{:g}/{:g}\n'.format('m', NUM_TRAIN_SAMPS, M))
	fid.write('{:<20}{:g}\n\n'.format('anomalous', numPtsIgnored))

	fid.write('{:<20}{:<20}{:<20}{:<20}\n\n'.format('Metric', 'MAGIC-LAT', 'GPR', 'quLATi'))
	fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('NMSE', nmse, nmseGPR, nmsequLATi))
	fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('MAE', mae, maeGPR, maequLATi))
	fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('DeltaE', dE, dEGPR, dEquLATi))

print('Success.\n')
print('Results saved to ' + outSubDir + '\n')