import time

import os

import numpy as np
import math
import random

# plotting packages
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import cv2

# Gaussian process regression interpolation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


from utils import *
from const import *
from magicLAT import *

from pygsp import graphs, filters, plotting

"""
p033 = 9
p034 = 14
p035 = 18
p037 = 20
"""
PATIENT_MAP				=		20

NUM_TRAIN_SAMPS 		= 		800
EDGE_THRESHOLD			=		50

""" Read the files """
meshFile = meshNames[PATIENT_MAP]
latFile = latNames[PATIENT_MAP]
nm = meshFile[0:-5]
patient = nm[7:10]

print('Reading files for ' + nm + ' ...\n')
[coordinateMatrix, connectivityMatrix] = readMesh(os.path.join(dataDir, meshFile))
[allLatCoord, allLatVal] = readLAT(os.path.join(dataDir, latFile))

n = len(coordinateMatrix)

mapIdx = [i for i in range(n)]
mapCoord = [coordinateMatrix[i] for i in mapIdx]

allLatIdx, allLatCoord, allLatVal = mapSamps(mapIdx, mapCoord, allLatCoord, allLatVal)

M = len(allLatIdx)

# KD Tree to find the nearest mesh vertex
k = 6
coordKDtree = cKDTree(allLatCoord)
[dist, nearestVers] = coordKDtree.query(allLatCoord, k=k)

anomalous = np.zeros(M)

# for i in range(M):
# 	verCoord = allLatCoord[i]
# 	verVal = allLatVal[i]

# 	neighbors = [nearestVers[i, n] for n in range(1,k) if dist[i,n] < 10]

# 	cnt = 0
# 	for neighVer in neighbors:
# 		neighVal = allLatVal[neighVer]

# 		if abs(verVal - neighVal) > 50:
# 			cnt += 1
# 		else:
# 			break

# 	if cnt == len(neighbors):
# 		anomalous[i] = 1

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
	[EDGES, TRI] = edgeMatrix(coordinateMatrix, connectivityMatrix)

	print('Writing edge matrix to file...')
	with open(edgeFile, 'wb') as fid:
		np.save(fid, EDGES)
else:
	EDGES = np.load(edgeFile, allow_pickle=True)

G = graphs.Graph(getUnWeightedAdj(n, EDGES), coords=coordinateMatrix)
G.estimate_lmax()

if not os.path.isdir('results_wavelet'):
	os.makedirs('results_wavelet')

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




g = filters.Abspline(G, Nf=10)

synthWavIdx = [0, 1, 2, 3, 4, 5]
g1 = filters.Abspline(G, Nf=len(synthWavIdx))

lat = np.zeros((n,1))
for i in range(len(TrIdx)):
	verIdx = TrIdx[i]
	lat[verIdx] = TrVal[i]

# G.plot_signal(lat)

sf = g.filter(lat)

# G.plot_signal(sf[:,0], backend='matplotlib')
# G.plot_signal(sf[:,1], backend='matplotlib')
# G.plot_signal(sf[:,2], backend='matplotlib')

latEst = g1.filter(sf[:,synthWavIdx])

# G.plot_signal(latEst)

est = latEst[TstIdx]
true = TstVal
nmse = np.sum(abs(est - true)**2)/np.sum((true - np.mean(true))**2)
print(nmse)

# plt.show()

""" Figure parameters """
# For colorbar ranges
MINLAT = math.floor(min(latVals)/10)*10
MAXLAT = math.ceil(max(latVals)/10)*10

triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

""" Figure 2 of the entire estimated signals """
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()

# Plot 1: Ground truth (front)
thisAx = axes[0]

pltSig = latVals
pltCoord = np.array(latCoords)

elev = 24
azim = -135

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

thisAx.set_title('Ground truth (front)')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

# Plot 2: MAGIC-LAT output signal (front)
thisAx = axes[1]

pltSig = latEst
pltCoord = coordinateMatrix

elev = 24
azim = -135

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

thisAx.set_title('WMAGIC-LAT (front)')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

plt.show()