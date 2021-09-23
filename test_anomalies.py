
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
from vedo import Mesh, Points, Point, show, Plotter

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


import utils
from const import DATADIR, DATAFILES



OUTDIR				 	=		'test_anomalies_results'


""" Parse the input for data index argument. """
parser = argparse.ArgumentParser(
    description='Processes a single mesh file for comparison of MAGIC-LAT, GPR, and quLATi performance.')

parser.add_argument('-i', '--idx', required=True, default='11',
                    help='Data index to process. \
                    Default: 11')

args = parser.parse_args()

PATIENT_IDX				=		int(vars(args)['idx'])

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

# For colorbar ranges
MINLAT = math.floor(min(allLatVal)/10)*10
MAXLAT = math.ceil(max(allLatVal)/10)*10

allPoints = Points(allLatCoord, r=10).cmap('rainbow_r', allLatVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')

vplt = Plotter(N=1, axes=0, interactive=True)
for i in range(M):
	idx = allLatIdx[i]
	coord = allLatCoord[i]
	val = allLatVal[i]

	testPoint = Point(coord, r=20).cmap('rainbow_r', [val], vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')
	vplt.show(mesh, allPoints, testPoint, title='i={:g}/{:g}'.format(i, M), bg='black')

vplt.close()