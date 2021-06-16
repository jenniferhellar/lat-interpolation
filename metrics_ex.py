
"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""

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

NUM_TRAIN_SAMPS 		= 		30
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



sCoord = [mapCoord[i] for i in mapIdx if (mapCoord[i][0] < 0 and mapCoord[i][1] < 60 and (float(mapCoord[i][0]) + 5/4*float(mapCoord[i][1])) < 20 and mapCoord[i][2] > 151 and mapCoord[i][2] < 180)]
sOrigIdx = [i for i in mapIdx if (mapCoord[i][0] < 0 and mapCoord[i][1] < 60 and (float(mapCoord[i][0]) + 5/4*float(mapCoord[i][1])) < 20 and mapCoord[i][2] > 151 and mapCoord[i][2] < 180)]
sN = len(sCoord)
sIdx = [i for i in range(sN)]

S_IS_SAMP = [False for i in range(sN)]
sLAT = [0 for i in range(sN)]
for i in range(sN):
	verIdx = sIdx[i]
	verOrigIdx = sOrigIdx[i]
	if (verOrigIdx in latIdx):
		S_IS_SAMP[verIdx] = True
		sLAT[verIdx] = mapLAT[verOrigIdx]

sLatIdx = [sIdx[i] for i in range(sN) if S_IS_SAMP[i] is True]
sLatCoord = [sCoord[i] for i in range(sN) if S_IS_SAMP[i] is True]
sLatVal = [sLAT[i] for i in range(sN) if S_IS_SAMP[i] is True]

sM = len(sLatIdx)

sVertices = np.array(sCoord)
sFaces = []

for tri in faces:
	idx0 = int(tri[0])
	idx1 = int(tri[1])
	idx2 = int(tri[2])

	if (idx0 in sOrigIdx) and (idx1 in sOrigIdx) and (idx2 in sOrigIdx):
		s_idx0 = sIdx[sOrigIdx.index(idx0)]
		s_idx1 = sIdx[sOrigIdx.index(idx1)]
		s_idx2 = sIdx[sOrigIdx.index(idx2)]
		S_tri = [s_idx0, s_idx1, s_idx2]
		sFaces.append(S_tri)

edgeFile = 'E_p{}_section.npy'.format(patient)
if not os.path.isfile(edgeFile):
	[EDGES, TRI] = edgeMatrix(sVertices, sFaces)

	print('Writing edge matrix to file...')
	with open(edgeFile, 'wb') as fid:
		np.save(fid, EDGES)
else:
	EDGES = np.load(edgeFile, allow_pickle=True)

print('{:<20}{:g}'.format('n', sN))
print('{:<20}{:g}/{:g}'.format('m', NUM_TRAIN_SAMPS, sM))


""" Do the interpolation """

sampLst = [i for i in range(sM)]

tr_i = random.sample(sampLst, NUM_TRAIN_SAMPS)
tst_i = [i for i in sampLst if i not in tr_i]

# get vertex indices of labelled/unlabelled nodes
TrIdx = sorted(np.take(sLatIdx, tr_i))
TstIdx = sorted(np.take(sLatIdx, tst_i))

# get vertex coordinates
TrCoord = [sCoord[i] for i in TrIdx]
TstCoord = [sCoord[i] for i in TstIdx]

# get mapLAT signal values
TrVal = [sLAT[i] for i in TrIdx]
TstVal = [sLAT[i] for i in TstIdx]


""" MAGIC-LAT estimate """
latEst = magicLAT(sVertices, sFaces, EDGES, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)


""" Error metrics """
nmse = calcNMSE(TstVal, latEst[TstIdx])
mae = calcMAE(TstVal, latEst[TstIdx])
print(nmse)
print(mae)


mesh = Mesh([sVertices, sFaces])
# mesh.backColor('white').lineColor('black').lineWidth(0.25)
mesh.c('grey')

# For colorbar ranges
MINLAT = math.floor(min(allLatVal)/10)*10
MAXLAT = math.ceil(max(allLatVal)/10)*10

elev = 0
azim = 120
roll = -45

# # Plot 0: Ground truth
# trueSigPoints = Points(sLatCoord, r=10).cmap('rainbow_r', sLatVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
# show(mesh, trueSigPoints, title='All raw points', axes=9).close()


plt = Plotter(N=3, axes=9)

# Plot 0: Ground truth
trueSigPoints = Points(sLatCoord, r=10).cmap('rainbow_r', sLatVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
plt.show(mesh, trueSigPoints, 'all known points', azimuth=azim, elevation=elev, roll=roll, at=0)

# Plot 1: Training points
trainPoints = Points(TrCoord, r=10).cmap('rainbow_r', TrVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
plt.show(mesh, trainPoints, 'training points', at=1)

# Plot 2: MAGIC-LAT output signal
pts = Points(sVertices, r=10).cmap('rainbow_r', latEst, vmin=MINLAT, vmax=MAXLAT).addScalarBar()

# mesh.interpolateDataFrom(pts, N=1).cmap('rainbow_r').addScalarBar()

plt.show(mesh, pts, 'interpolation result', title='MAGIC-LAT', at=2, interactive=True)

plt.screenshot(filename=os.path.join(outDir, 'metrics_ex.png'), scale=2, returnNumpy=False)
plt.close()
