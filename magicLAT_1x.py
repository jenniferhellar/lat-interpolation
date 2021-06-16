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

M = len(latIdx)

print('{:<20}{:g}'.format('n', n))
print('{:<20}{:g}/{:g}'.format('m', NUM_TRAIN_SAMPS, M))
print('{:<20}{:g}'.format('ignored', numPtsIgnored))
# exit()

mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]

edgeFile = 'E_p{}.npy'.format(patient)
if not os.path.isfile(edgeFile):
	[EDGES, TRI] = edgeMatrix(vertices, faces)

	print('Writing edge matrix to file...')
	with open(edgeFile, 'wb') as fid:
		np.save(fid, EDGES)
else:
	EDGES = np.load(edgeFile, allow_pickle=True)

if not os.path.isdir(outDir):
	os.makedirs(outDir)

sampLst = [i for i in range(M)]


tr_i = random.sample(sampLst, NUM_TRAIN_SAMPS)
tst_i = [i for i in sampLst if i not in tr_i]

# get vertex indices of labelled/unlabelled nodes
TrIdx = sorted(np.take(latIdx, tr_i))
TstIdx = sorted(np.take(latIdx, tst_i))

# get vertex coordinates
TrCoord = [mapCoord[i] for i in TrIdx]
TstCoord = [mapCoord[i] for i in TstIdx]

# get mapLAT signal values
TrVal = [mapLAT[i] for i in TrIdx]
TstVal = [mapLAT[i] for i in TstIdx]


""" MAGIC-LAT estimate """
latEst = magicLAT(vertices, faces, EDGES, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)


""" Error metrics """
nmse = calcNMSE(TstVal, latEst[TstIdx])


""" Figure parameters """
# For colorbar ranges
MINLAT = math.floor(min(allLatVal)/10)*10
MAXLAT = math.ceil(max(allLatVal)/10)*10

# triang = mtri.Triangulation(vertices[:,0], vertices[:,1], triangles=faces)


""" Figure 1 of the Training and Test points """

# Plot 1: Ground truth test points

testPoints = Points(TstCoord, r=10).cmap('rainbow_r', TstVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
# show(mesh, testPoints, __doc__, axes=9).close()
# show(mesh, testPoints, title='Ground truth (test points)', axes=9).close()

# Plot 2: Training points
# show(mesh, trainPoints, title='Training points', axes=9).close()

# Plot 3: MAGIC-LAT output test points
testOutPoints = Points(TstCoord, r=10).cmap('rainbow_r', latEst[TstIdx], vmin=MINLAT, vmax=MAXLAT).addScalarBar()
# show(mesh, testOutPoints, title='MAGIC-LAT Estimate, NMSE = {:.4f}'.format(nmse), axes=9).close()



""" Figure 2 of the entire estimated signals """
plt = Plotter(N=3, axes=9)

# Plot 0: Ground truth
trueSigPoints = Points(latCoords, r=15).cmap('rainbow_r', latVals, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
plt.show(mesh, trueSigPoints, title='All raw points', at=0)

# Plot 1:
trainPoints = Points(TrCoord, r=10).cmap('rainbow_r', TrVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
plt.show(mesh, trainPoints, title='Training points', at=1)

# Plot 2: MAGIC-LAT output signal
# whitePoints = Points(latCoords, r = 11)
pts = Points(vertices, r=8).cmap('rainbow_r', latEst, vmin=MINLAT, vmax=MAXLAT).addScalarBar()

# mesh.interpolateDataFrom(pts, N=4).cmap('rainbow_r').addScalarBar()

# plt.show(mesh, testOutPoints, title='MAGIC-LAT', at=2, interactive=True).close()

plt.show(mesh, trueSigPoints, pts, title='MAGIC-LAT', at=2, interactive=True).close()
