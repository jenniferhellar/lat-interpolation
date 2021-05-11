
"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""
import os

import numpy as np
import math

# plotting packages
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# KD-Tree for mapping to nearest point
from scipy.spatial import cKDTree

# cross-validation package
from sklearn.model_selection import KFold

# nearest-neighbor interpolation
from scipy.interpolate import griddata

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


from utils import *
from const import *


# 20 is the one I have been working with
mapIdx = 20

""" Read the files """
meshFile = meshNames[mapIdx]
latFile = latNames[mapIdx]
nm = meshFile[0:-5]

print('Reading files for ' + nm + ' ...')
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[latCoords, latVals] = readLAT(dataDir + latFile)

N = len(coordinateMatrix)	# number of vertices in the graph
M = len(latCoords)			# number of signal samples

if N > 11000:
	print('Graph too large!')
	exit(0)

IDX = [i for i in range(N)]
COORD = [coordinateMatrix[i] for i in IDX]

# Map data points to mesh coordinates
coordKDtree = cKDTree(coordinateMatrix)
[dist, idxs] = coordKDtree.query(latCoords, k=1)

IS_SAMP = [False for i in range(N)]
LAT = [0 for i in range(N)]
for i in range(M):
	verIdx = idxs[i]
	IS_SAMP[verIdx] = True
	LAT[verIdx] = latVals[i]

SAMP_IDX = [IDX[i] for i in range(N) if IS_SAMP[i] is True]
SAMP_COORD = [COORD[i] for i in range(N) if IS_SAMP[i] is True]
SAMP_LAT = [LAT[i] for i in range(N) if IS_SAMP[i] is True]

M = len(SAMP_IDX)

# For colorbar ranges
MINLAT = math.floor(min(SAMP_LAT)/10)*10
MAXLAT = math.ceil(max(SAMP_LAT)/10)*10

# Figure view
elev = 24
azim = -135

# print(len(latCoords), len(coordinateMatrix), len(connectivityMatrix))
# print(max(latVals), min(latVals))

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()

triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

ax = axes[0]
# plot the graph/manifold
triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)
ax.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
# ax.plot_trisurf(triang, coordinateMatrix[:,2], color=(0,0,0,0), edgecolor='k', linewidth=.05)

# plot the signal
pos = ax.scatter(latCoords[:,0], latCoords[:,1], latCoords[:,2], c=latVals, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 20)

ax.set_title('Original LAT Coordinates')
ax.set_xlabel('X', fontweight ='bold') 
ax.set_ylabel('Y', fontweight ='bold') 
ax.set_zlabel('Z', fontweight ='bold')
ax.view_init(elev, azim)

ax = axes[1]
ax.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pltCoord = np.array(SAMP_COORD)
pltSig = SAMP_LAT
pos = ax.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 20)
ax.set_title('LAT Value Assigned to Nearest Mesh Vertex')
ax.set_xlabel('X', fontweight ='bold') 
ax.set_ylabel('Y', fontweight ='bold') 
ax.set_zlabel('Z', fontweight ='bold')
ax.view_init(elev, azim)

cax = fig.add_axes([ax.get_position().x1+0.03,ax.get_position().y0,0.01,ax.get_position().height])
plt.colorbar(pos, cax=cax, label='LAT (ms)') # Similar to fig.colorbar(im, cax = cax)

plt.show()