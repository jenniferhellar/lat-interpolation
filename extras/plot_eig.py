
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


LOAD_EIGEN			=		1

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

# Assign unknown coordinates an initial value of the nearest known point
UNKNOWN = [COORD[i] for i in range(N) if IS_SAMP[i] is False]
UNKNOWN = griddata(np.array(SAMP_COORD), np.array(SAMP_LAT), np.array(UNKNOWN), method='nearest')
currIdx = 0
for i in range(N):
	if IS_SAMP[i] is False:
		LAT[i] = UNKNOWN[currIdx]
		currIdx += 1

# For colorbar ranges
MINLAT = math.floor(min(SAMP_LAT)/10)*10
MAXLAT = math.ceil(max(SAMP_LAT)/10)*10

# Figure view
elev = 24
azim = -135

if not LOAD_EIGEN:
	print('Generating edge matrix ...')
	threshold = 50
	[EDGES, TRI, excl_midpt] = getE(coordinateMatrix, connectivityMatrix, LAT, threshold)
	print('Edges excluded: ', str(len(excl_midpt)))

	print('Calculating Laplacian matrix ...')

	A = getUnWeightedAdj(coordinateMatrix, EDGES, TRI)
	D = np.diag(A.sum(axis=1))
	# L = D - A

	# W = getAdjMatrixCotan(coordinateMatrix, EDGES, TRI)
	W = getAdjMatrixExp(coordinateMatrix, EDGES, TRI)
	L = D - W

	# pltAdjMatrix(W, 0, 20, 'W(i,j) = 1/d(i,j)')
	# pltAdjMatrix(W1, 0, 20, 'Cotangent Weights')

	print('Calculating eigendecomposition ...')
	lambdas, U = np.linalg.eigh(L)
	V = np.diag(lambdas)

	print('Writing V to file...')
	fid = open('V.npy', 'wb')
	np.save(fid, V)
	fid.close()

	print('Writing U to file...')
	fid = open('U.npy', 'wb')
	np.save(fid, U)
	fid.close()

else:
	V = np.load('V.npy')
	U = np.load('U.npy')


pltThresh = 0.005
vec0 = N-1
vec1 = N-2
vec2 = N-3
vec3 = N-4
vec4 = N-5
vec5 = N-6

triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()


""" Plot 0 """
thisAx = axes[0]

pltSig = U[:, vec0]
pltI = [i for i in range(N) if abs(pltSig[i]) > pltThresh]
pltSig = pltSig[pltI]
pltCoord = np.array([COORD[i] for i in pltI])

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color=(0,0,0,0), edgecolor='k', linewidth=.05)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=-1, vmax=1, s = 20)

titleA = '{:g}'.format((N-1) - vec0)

thisAx.set_title(r'$U_{'+titleA+'}$')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)

# cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
# plt.colorbar(pos, cax=cax, label='magnitude', orientation="horizontal")


""" Plot 1 """
thisAx = axes[1]

pltSig = U[:, vec1]
pltI = [i for i in range(N) if abs(pltSig[i]) > pltThresh]
pltSig = pltSig[pltI]
pltCoord = np.array([COORD[i] for i in pltI])

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color=(0,0,0,0), edgecolor='k', linewidth=.05)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=-1, vmax=1, s = 20)

titleB = '{:g}'.format((N-1) - vec1)

thisAx.set_title(r'$U_{'+titleB+'}$')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)

# cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
# plt.colorbar(pos, cax=cax, label='magnitude', orientation="horizontal")


""" Plot 2 """
thisAx = axes[2]

pltSig = U[:, vec2]
pltI = [i for i in range(N) if abs(pltSig[i]) > pltThresh]
pltSig = pltSig[pltI]
pltCoord = np.array([COORD[i] for i in pltI])

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color=(0,0,0,0), edgecolor='k', linewidth=.05)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=-1, vmax=1, s = 20)

titleB = '{:g}'.format((N-1) - vec2)

thisAx.set_title(r'$U_{'+titleB+'}$')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)

# cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
# plt.colorbar(pos, cax=cax, label='magnitude', orientation="horizontal")


""" Plot 3 """
thisAx = axes[3]

pltSig = U[:, vec3]
pltI = [i for i in range(N) if abs(pltSig[i]) > pltThresh]
pltSig = pltSig[pltI]
pltCoord = np.array([COORD[i] for i in pltI])

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color=(0,0,0,0), edgecolor='k', linewidth=.05)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=-1, vmax=1, s = 20)

titleB = '{:g}'.format((N-1) - vec3)

thisAx.set_title(r'$U_{'+titleB+'}$')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)

# cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
# plt.colorbar(pos, cax=cax, label='magnitude', orientation="horizontal")


""" Plot 4 """
thisAx = axes[4]

pltSig = U[:, vec4]
pltI = [i for i in range(N) if abs(pltSig[i]) > pltThresh]
pltSig = pltSig[pltI]
pltCoord = np.array([COORD[i] for i in pltI])

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color=(0,0,0,0), edgecolor='k', linewidth=.05)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=-1, vmax=1, s = 20)

titleB = '{:g}'.format((N-1) - vec4)

thisAx.set_title(r'$U_{'+titleB+'}$')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)

cax = fig.add_axes([thisAx.get_position().x0+0.005,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
plt.colorbar(pos, cax=cax, label='magnitude', orientation="horizontal")


""" Plot 5 """
thisAx = axes[5]

pltSig = U[:, vec5]
pltI = [i for i in range(N) if abs(pltSig[i]) > pltThresh]
pltSig = pltSig[pltI]
pltCoord = np.array([COORD[i] for i in pltI])

thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color=(0,0,0,0), edgecolor='k', linewidth=.05)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=-1, vmax=1, s = 20)

titleB = '{:g}'.format((N-1) - vec5)

thisAx.set_title(r'$U_{'+titleB+'}$')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)

# cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
# plt.colorbar(pos, cax=cax, label='magnitude', orientation="horizontal")


plt.show()