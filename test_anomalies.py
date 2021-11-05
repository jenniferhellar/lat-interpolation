"""
--------------------------------------------------------------------------------
Plots individual LAT observations for manual inspection and identification of
anomalous points.
--------------------------------------------------------------------------------

usage: test_anomalies.py [-h] -i IDX

Sequentially plots LAT observations for manual inspection.

optional arguments:
  -h, --help            show this help message and exit
  -i IDX, --idx IDX     Data index to process. Default: 11

DATA INDICES:
	  p033 = 4 (3-RV-FAM-PVC-A-NORMAL), 5 (4-RV-FAM-PVC-A-LAT-HYBRID)
		p034 = 6 (4-RVFAM-LAT-HYBRID), 7 (5-RVFAM-PVC), 8 (6-RVFAM-SINUS-VOLTAGE)
		p035 = 9 (8-SINUS)
		p037 = 11 (9-RV-SINUS-VOLTAGE)

Requirements: 
	os, argparse,
	numpy, math, random, 
	vedo

File: test_anomalies.py

Author: Jennifer Hellar
Email: jenniferhellar@gmail.com
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


""" Parse the input for data index argument. """
parser = argparse.ArgumentParser(
    description='Sequentially plots LAT observations for manual inspection.')

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

""" Read the files """
print('\nProcessing ' + nm + ' ...\n')
[vertices, faces] = readMesh(os.path.join(DATADIR, meshFile))
[OrigLatCoords, OrigLatVals] = readLAT(os.path.join(DATADIR, latFile))

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

allPoints = Points(allLatCoord, r=10).cmap('gist_rainbow', allLatVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')

vplt = Plotter(N=1, axes=0, interactive=True)
for i in range(M):
	idx = allLatIdx[i]
	coord = allLatCoord[i]
	val = allLatVal[i]

	# plot current point as larger than the others
	testPoint = Point(coord, r=20).cmap('gist_rainbow', [val], vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white')
	vplt.show(mesh, allPoints, testPoint, title='i={:g}/{:g}'.format(i, M), bg='black')

vplt.close()