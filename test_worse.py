
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
from vedo import Mesh, Points, Plotter, Point, Text2D

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

OUTDIR				 	=		'test_worse_results'


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
	# anomalous = utils.isAnomalous(allLatCoord, allLatVal)
	if PATIENT_IDX == 4:
		anomIdx = [25, 112, 159, 218, 240, 242, 264]
	elif PATIENT_IDX == 5:
		anomIdx = [119, 150, 166, 179, 188, 191, 209, 238]
	elif PATIENT_IDX == 6:
		anomIdx = [11, 12, 41, 43, 56, 59, 63, 78, 91, 95, 101, 120, 127, 156, 158, 160, 177, 183]
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
	Figure 2: Ground truth (entire), training points, and quLATi (entire)
	"""
	utils.plotSaveEntire(mesh, latCoords, latVals, TrCoord, TrVal, latEstquLATi, 
		azimuth, elev, roll, MINLAT, MAXLAT,
		outSubDir, title='quLATi', filename='quLATi', ablFile=ablFile)


"""
Error metrics
"""

dE = metrics.deltaE(TstVal, latEst[TstIdx], MINLAT, MAXLAT)
dEquLATi = metrics.deltaE(TstVal, latEstquLATi[TstIdx], MINLAT, MAXLAT)

if verbose:
	print('\tWriting to file...')

with open(os.path.join(outSubDir, 'metrics.txt'), 'w') as fid:
	fid.write(nm + '\n\n')
	fid.write('{:<20}{:g}\n'.format('n', n))
	fid.write('{:<20}{:g}/{:g}\n'.format('m', NUM_TRAIN_SAMPS, M))
	fid.write('{:<20}{:g}\n\n'.format('anomalous', numPtsIgnored))

	fid.write('{:<20}{:<20}{:<20}\n\n'.format('', 'MAGIC-LAT', 'quLATi'))
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('DeltaE', np.mean(dE), np.mean(dEquLATi)))

print('Success.\n')
print('Results saved to ' + outSubDir + '\n')

# if np.mean(dE) < np.mean(dEquLATi):
# 	print('\nMAGIC-LAT better this run.  Try again.')
# 	exit(0)


dE = dE.flatten()
dEquLATi = dEquLATi.flatten()

l = len(TstIdx)
worseTstIdx = [dE[i] > dEquLATi[i] for i in range(l)]
print('\nMAGIC-LAT worse for {:g}/{:g} test points'.format(np.sum(worseTstIdx), len(TstIdx)))

worseIdx = [TstIdx[i] for i in range(l) if worseTstIdx[i] > 0]
worseCoord = [TstCoord[i] for i in range(l) if worseTstIdx[i] > 0]
worseVal = [latEst[TstIdx][i] for i in range(l) if worseTstIdx[i] > 0]
quLATiVal = [latEstquLATi[TstIdx][i] for i in range(l) if worseTstIdx[i] > 0]

TrPoints = Points(TrCoord, r=10).cmap('rainbow_r', TrVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')

vplt = Plotter(N=3, axes=0)
for i in range(len(TstIdx)):
	if dE[i] > dEquLATi[i]:
		idx = TstIdx[i]
		coord = vertices[idx]
		val = np.array(latEst[idx]).flatten()[0]
		quval = latEstquLATi[idx]
		trueval = TstVal[i]

		print(idx, coord, val, quval)

		testPoint = Point(coord, r=20).cmap('rainbow_r', [trueval], vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')
		vplt.show(mesh, TrPoints, testPoint, Text2D(txt='i={:g}, true'.format(idx), pos='top-left', c='white'), bg='black', at = 0, interactive=False)

		testPoint = Point(coord, r=20).cmap('rainbow_r', [val], vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')
		vplt.show(mesh, TrPoints, testPoint, Text2D(txt='i={:g}, MAGIC-LAT, dE={:.6f}'.format(idx, dE[i]), pos='top-left', c='white'), bg='black', at = 1)

		testPoint = Point(coord, r=20).cmap('rainbow_r', [quval], vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')
		vplt.show(mesh, TrPoints, testPoint, Text2D(txt='i={:g}, quLATi, dE={:.6f}'.format(idx, dEquLATi[i]), pos='top-left', c='white'), bg='black', at = 2, interactive=True)

		vplt.clear(at=0)
		vplt.clear(at=1)
		vplt.clear(at=2)

vplt.close()
