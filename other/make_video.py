"""
--------------------------------------------------------------------------------
Manifold Approximating Graph Interpolation on Cardiac mapLAT data (MAGIC-LAT).
--------------------------------------------------------------------------------

Description: Cross-validation to randomly select test sets for interpolation.
5x repetitition for error mean and variance estimation.

Requirements: os, numpy, matplotlib, sklearn, scipy, math

File:

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""

import time

import os

import numpy as np
import math
import random

# plotting packages
from vedo import *

import cv2

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
PATIENT_MAP				=		21

NUM_TRAIN_SAMPS 		= 		300
EDGE_THRESHOLD			=		50

outDir				 	=		'results_wip'

""" Read the files """
meshFile = meshNames[PATIENT_MAP]
latFile = latNames[PATIENT_MAP]
nm = meshFile[0:-5]
patient = nm[7:10]

print('Reading files for ' + nm + ' ...\n')
[vertices, faces] = readMesh(os.path.join(dataDir, meshFile))
[OrigLatCoords, OrigLatVals] = readLAT(os.path.join(dataDir, latFile))

n = len(vertices)

mapIdx = [i for i in range(n)]
mapCoord = [vertices[i] for i in mapIdx]

allLatIdx, allLatCoord, allLatVal = mapSamps(mapIdx, mapCoord, OrigLatCoords, OrigLatVals)

M = len(allLatIdx)

mesh = Mesh([vertices, faces])
# mesh.backColor('white').lineColor('black').lineWidth(0.25)
mesh.c('grey')

origLatPoints = Points(OrigLatCoords, r=10).cmap('rainbow_r', OrigLatVals, vmin=np.min(OrigLatVals), vmax=np.max(OrigLatVals)).addScalarBar()
latPoints = Points(allLatCoord, r=10).cmap('rainbow_r', allLatVal, vmin=np.min(allLatVal), vmax=np.max(allLatVal)).addScalarBar()

# KD Tree to find the nearest mesh vertex
k = 6
coordKDtree = cKDTree(allLatCoord)
[dist, nearestVers] = coordKDtree.query(allLatCoord, k=k)

anomalous = np.zeros(M)

for i in range(M):
	verCoord = allLatCoord[i]
	verVal = allLatVal[i]

	neighbors = [nearestVers[i, n] for n in range(1,k) if dist[i,n] < 5]

	adj = len(neighbors)

	cnt = 0
	for neighVer in neighbors:
		neighVal = allLatVal[neighVer]

		if abs(verVal - neighVal) > 50:
			cnt += 1
		else:
			break

	# if (cnt >= (len(neighbors)-1) and len(neighbors) > 1):	# differs from all but 1 neighbor by >50ms and has at least 2 neighbors w/in 5mm
	if cnt > 1 and adj > 1:
		anomalous[i] = 1
		# print(cnt, adj)

		# print(verVal, [allLatVal[neighVer] for neighVer in neighbors])

numPtsIgnored = np.sum(anomalous)

latIdx = [allLatIdx[i] for i in range(M) if anomalous[i] == 0]
latCoords = [allLatCoord[i] for i in range(M) if anomalous[i] == 0]
latVals = [allLatVal[i] for i in range(M) if anomalous[i] == 0]


if not os.path.isdir(outDir):
	os.makedirs(outDir)

""" Figure parameters """
# For colorbar ranges
MINLAT = math.floor(min(latVals)/10)*10
MAXLAT = math.ceil(max(latVals)/10)*10

plt = Plotter(N=1, axes=4, offscreen=True)

# Plot 0: Ground truth
trueSigPoints = Points(latCoords, r=15).cmap('rainbow_r', latVals, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
plt.show(mesh, trueSigPoints, title='All raw points', at=0)

video = Video("mesh.mp4", duration=12, backend='opencv') # backend='opencv'
video.action(azimuth_range=(0,359))
video.close()
