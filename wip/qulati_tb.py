from qulati import gpmi, gpmi_rr, eigensolver

import time

import os

import numpy as np
import math
import random

# plotting packages
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


from utils import *
from const import *

"""
p033 = 9
p034 = 14
p035 = 18
p037 = 20
"""
PATIENT_MAP				=		20

NUM_TRAIN_SAMPS 		= 		300
EDGE_THRESHOLD			=		50

""" Read the files """
meshFile = meshNames[PATIENT_MAP]
latFile = latNames[PATIENT_MAP]
nm = meshFile[0:-5]
patient = nm[7:10]

print('Reading files for ' + nm + ' ...\n')
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[allLatCoord, allLatVal] = readLAT(dataDir + latFile)

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

for i in range(M):
	verCoord = allLatCoord[i]
	verVal = allLatVal[i]

	neighbors = [nearestVers[i, n] for n in range(1,k) if dist[i,n] < 10]

	cnt = 0
	for neighVer in neighbors:
		neighVal = allLatVal[neighVer]

		if abs(verVal - neighVal) > 50:
			cnt += 1
		else:
			break

	if cnt == len(neighbors):
		anomalous[i] = 1

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

X = coordinateMatrix
Tri = np.array(connectivityMatrix)

"""
Solve the eigenproblem

For a mesh with 5 holes (4 pulmonary veins and 1 mitral valve), where 10 layers of mesh elements are to be appeneded to the edges representing these holes, the Laplacian eigenvalue problem can be solved for the 256 smallest eigenvalues with:
""" 

Q, V, gradV, centroids = eigensolver(X, Tri, holes = 0, layers = 10, num = 256)

"""
`Q` are the 256 smallest eigenvalues, and 
`V` are the corresponding eigenfunction values at vertex and centroid locations. 
The gradient of the eigenfunctions at face `centroids` are given by `gradV`.

Note that this problem only needs solving a single time given a specific mesh, so results ought to be saved for future use.

The class for doing the interpolation can then be intialized with
"""

# model with reduced rank efficiency
model = gpmi_rr.Matern(X, Tri, Q, V, gradV)

# # slower, but more general modelling is possible
# model = gpmi.Matern(X, Tri, Q, V, gradV)

""" Set the data

For observations (preferably standardized) defined at vertices and centroids, data can be set using

vertices` is a zero indexed integer array referencing which vertex or face centroid an observation belongs to. 
Observations can be assigned to vertices with indices `0:X.shape[0]` and assigned to face centroids with indices `X.shape[0] + Tri_index` (where `Tri_index` refers to faces defined in `Tri`).
"""
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

obs = TrVal
vertices = TrIdx

# when using gpmi_rr model
model.set_data(obs, vertices)

# when using gpmi model
#model.set_data(obs, obs_std, vertices)

"""
For the Matern kernel class, the kernel smoothness and explicit mean function must both be set:
"""

model.kernelSetup(smoothness = 3./2.)
model.setMeanFunction() # zero for gpmi.Matern class

# optimize the nugget
model.optimize(nugget = None, restarts = 5)


""" Predictions

Predictions at vertices and centroids can be obtained with the following, where the 
posterior mean and standard deviation are returned.
(vertex predictions are indexed `0:X.shape[0]`, 
centroid predictions are indexed `X.shape[0]:(X.shape[0] + Tri.shape[0]`).
"""

pred_mean, pred_stdev = model.posterior()

latEst = pred_mean[0:X.shape[0]]


""" Error metrics """
nmse = calcNMSE(TstVal, latEst[TstIdx])



""" Figure parameters """
# For colorbar ranges
MINLAT = math.floor(min(latVals)/10)*10
MAXLAT = math.ceil(max(latVals)/10)*10

triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)


""" Figure 1 of the Training and Test points """
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()

# Plot 1: Ground truth test points
thisAx = axes[0]

pltSig = TstVal
pltCoord = np.array(TstCoord)

elev = 24
azim = -135

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 10)

thisAx.set_title('Ground truth (test points)\n')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)

# Plot 2: Training points
thisAx = axes[1]

pltSig = TrVal
pltCoord = np.array(TrCoord)

elev = 24
azim = -135

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 10)

thisAx.set_title('Training points')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)

# Plot 3: output test points
thisAx = axes[2]

pltSig = latEst[TstIdx]
pltCoord = np.array(TstCoord)

elev = 24
azim = -135

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 10)

thisAx.set_title('quLATi test points\nNMSE = {:.4f}'.format(nmse))
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

# Plot 4: output signal
thisAx = axes[3]

pltSig = latEst
pltCoord = coordinateMatrix

elev = 24
azim = -135

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

thisAx.set_title('quLATi whole\n')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

# Save the figure
if not os.path.isdir('results_qulati'):
	os.makedirs('results_qulati')
filename = os.path.join('results_qulati', 'p{}_t{:g}_m{:g}.png'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS))
fig.savefig(filename)
plt.close()