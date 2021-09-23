
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

import os

import argparse

import numpy as np
import math
import random

# plotting packages
from vedo import Mesh, Points, Plotter

# Gaussian process regression interpolation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# nearest-neighbor interpolation
from scipy.interpolate import griddata

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


import utils
import metrics
from const import DATADIR, DATAFILES
import magicLAT

import quLATiHelper


NUM_TRAIN_SAMPS			=		100
EDGE_THRESHOLD			=		50

OUTDIR				 	=		'plotMagicProcess_results'


""" Parse the input for data index argument. """
parser = argparse.ArgumentParser(
    description='Processes a single mesh file for comparison of MAGIC-LAT, GPR, and quLATi performance.')

parser.add_argument('-i', '--idx', required=True, default='11',
                    help='Data index to process. \
                    Default: 11')

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

n = len(vertices)

mapIdx = [i for i in range(n)]
mapCoord = [vertices[i] for i in mapIdx]

# Map the LAT samples to nearest mesh vertices
allLatIdx, allLatCoord, allLatVal = utils.mapSamps(mapIdx, mapCoord, OrigLatCoords, OrigLatVals)

M = len(allLatIdx)

# Identify and exclude anomalous LAT samples
anomalous = utils.isAnomalous(allLatCoord, allLatVal)

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

# NN interpolation of unknown vertices
latNN = [0 for i in range(n)]
unknownCoord = [vertices[i] for i in range(n) if i not in TrIdx]
unknownCoord = griddata(np.array(TrCoord), np.array(TrVal), np.array(unknownCoord), method='nearest')
currIdx = 0
for i in range(n):
	if i not in TrIdx:
		latNN[i] = unknownCoord[currIdx]
		currIdx += 1
	else:
		latNN[i] = mapLAT[i]

updatedFaces = magicLAT.updateFaces(vertices, faces, latNN, TrCoord, EDGE_THRESHOLD)

latEst = magicLAT.magicLAT(vertices, faces, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)


mesh = Mesh([vertices, faces])
# mesh.backColor('white').lineColor('black').lineWidth(0.25)
mesh.c('grey')

verPoints = Points(vertices, r=10, c='white')
origLatPoints = Points(OrigLatCoords, r=10).cmap('rainbow_r', OrigLatVals, vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')
allLatPoints = Points(allLatCoord, r=10).cmap('rainbow_r', allLatVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')
latPoints = Points(latCoords, r=10).cmap('rainbow_r', latVals, vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')
trPoints = Points(TrCoord, r=10).cmap('rainbow_r', TrVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')
nnLatPoints = Points(vertices, r=10).cmap('rainbow_r', latNN, vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')
latEstPoints = Points(vertices, r=10).cmap('rainbow_r', latEst, vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')


a = 130
e = 0
r = 0
z = 1
verPoints = Points(vertices, r=5, c='white')

vplt = Plotter(N=1, axes=0, offscreen=True)
vplt.show(mesh, azimuth=a, elevation=e, roll=r, bg='black', zoom=z)
vplt.screenshot(filename=os.path.join(outSubDir, '0_mesh'), returnNumpy=False).close()

mesh.lineColor('white').lineWidth(1)
mesh.c('black')

vplt = Plotter(N=1, axes=0, offscreen=True)
vplt.show(mesh, verPoints, azimuth=a, elevation=e, roll=r, bg='black', zoom=z)
vplt.screenshot(filename=os.path.join(outSubDir, '1_graph'), returnNumpy=False).close()

vplt = Plotter(N=1, axes=0, offscreen=True)
vplt.show(mesh, allLatPoints, azimuth=a, elevation=e, roll=r, bg='black', zoom=z)
vplt.screenshot(filename=os.path.join(outSubDir, '2_graph_withAllLAT'), returnNumpy=False).close()

vplt = Plotter(N=1, axes=0, offscreen=True)
vplt.show(mesh, trPoints, azimuth=a, elevation=e, roll=r, bg='black', zoom=z)
vplt.screenshot(filename=os.path.join(outSubDir, '3_graph_withTrLAT'), returnNumpy=False).close()

vplt = Plotter(N=1, axes=0, offscreen=True)
vplt.show(mesh, nnLatPoints, azimuth=a, elevation=e, roll=r, bg='black', zoom=z)
vplt.screenshot(filename=os.path.join(outSubDir, '4_graph_withLATNN'), returnNumpy=False).close()

updatedMesh = Mesh([vertices, updatedFaces])
updatedMesh.lineColor('white').lineWidth(1)
updatedMesh.c('black')
updatedMesh.backColor('white')

vplt = Plotter(N=1, axes=0, offscreen=True)
vplt.show(updatedMesh, nnLatPoints, azimuth=a, elevation=e, roll=r, bg='black', zoom=z)
vplt.screenshot(filename=os.path.join(outSubDir, '5_updated_graph_zoomed_withLATNN'), returnNumpy=False).close()

z = 1
vplt = Plotter(N=1, axes=0, offscreen=True)
vplt.show(updatedMesh, trPoints, azimuth=a, elevation=e, roll=r, bg='black', zoom=z)
vplt.screenshot(filename=os.path.join(outSubDir, '6_updated_graph_withTrLAT'), returnNumpy=False).close()

vplt = Plotter(N=1, axes=0, offscreen=True)
vplt.show(updatedMesh, latEstPoints, azimuth=a, elevation=e, roll=r, bg='black', zoom=z)
vplt.screenshot(filename=os.path.join(outSubDir, '7_updated_graph_withLATest'), returnNumpy=False).close()

coloredMesh = Mesh([vertices, faces])
coloredMesh.interpolateDataFrom(latEstPoints, N=1).cmap('rainbow_r', vmin=MINLAT, vmax=MAXLAT).addScalarBar()

vplt = Plotter(N=1, axes=0, offscreen=True)
vplt.show(coloredMesh, trPoints, azimuth=a, elevation=e, roll=r, bg='black', zoom=z)
vplt.screenshot(filename=os.path.join(outSubDir, '8_mesh_interpolated_withTrLAT'), returnNumpy=False).close()

vplt = Plotter(N=1, axes=0, offscreen=True)
vplt.show(coloredMesh, latPoints, azimuth=a, elevation=e, roll=r, bg='black', zoom=z)
vplt.screenshot(filename=os.path.join(outSubDir, '9_mesh_interpolated_all'), returnNumpy=False).close()