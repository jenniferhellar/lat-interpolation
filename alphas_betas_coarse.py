"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""

import os

import numpy as np

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


dataDir = 'data/'
meshNames = ['mesh.mesh']
latNames = ['lat.txt']

i = 0

""" Read the files """
meshFile = meshNames[i]
latFile = latNames[i]
nm = meshFile[0:-5]

print('Reading files for ' + nm + ' ...')
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[latCoords, latVals] = readLAT(dataDir + latFile)

N = len(coordinateMatrix)	# number of vertices in the graph
M = len(latCoords)			# number of signal samples

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

"""
Get edges and corresponding adjacent triangles.

edges: list of edges, edge = set(v_i, v_j)
triangles: list of corr. triangles adj to each edge, tri = (v_i, v_j, v_k)
"""
print('Generating edge matrix ...')
[EDGES, TRI] = getE(connectivityMatrix)


""" 
A: NxN unweighted adjacency matrix
W: NxN weighted adjacency matrix (cotan weights)
D: NxN degree matrix
I: NxN identity matrix 

L: NxN Laplace matrix

"""
print('Calculating adjacency matrices ...')
A = getUnWeightedAdj(coordinateMatrix, EDGES, TRI)
# W = getAdjMatrix(coordinateMatrix, EDGES, TRI)

D = np.diag(A.sum(axis=1))
I = np.identity(N)

print('Calculating Laplacian matrix ...')
L = D - A


""" Hyperparameters """
alphas = [0.1, 1.0, 5.0, 10.0, 100.0, 1000.0]
betas = [0.1, 1.0, 5.0, 10.0, 100.0, 1000.0]

# alphas = [0.1, 1.0]
# betas = [0.1]

print()

if not os.path.isdir('res'):
	os.makedirs('res')

print('Writing alphas to file...')
fid = open(os.path.join('res','alphas.txt'), 'w')
np.array(alphas).tofile(fid, sep='\n', format='%.2f')
fid.close()

print('Writing betas to file...')
fid = open(os.path.join('res','betas.txt'), 'w')
np.array(betas).tofile(fid, sep='\n', format='%.2f')
fid.close()

""" Cross-validation """

folds = 10
kf12 = KFold(n_splits=folds, shuffle=True)

mse = np.zeros((len(alphas), len(betas)))
wmse = np.zeros((len(alphas), len(betas)))
snr = np.zeros((len(alphas), len(betas)))
wsnr = np.zeros((len(alphas), len(betas)))

for a_idx in range(len(alphas)):
	alpha = alphas[a_idx]

	for b_idx in range(len(betas)):
		beta = betas[b_idx]

		print('\nCross-validating alpha=' + str(alpha) + ', beta=' + str(beta))

		fold = 0

		yhat = np.zeros((N,1))

		for tr_i, tst_i in kf12.split(SAMP_IDX):

			print('\tFold ' + str(fold))

			y = np.zeros((N,1))
			
			M_l = np.zeros((N,N))
			M_u = np.zeros((N,N))

			# number of labelled and unlabelled vertices in this fold
			trLen = len(tr_i)
			tstLen = len(tst_i)

			# get vertex indices of labelled/unlabelled nodes
			TrIdx = sorted(np.take(SAMP_IDX, tr_i))
			TstIdx = sorted(np.take(SAMP_IDX, tst_i))

			# Compute graph interpolation estimate for unlabelled vertices in this fold
			for i in range(N):
				if i in TrIdx:
					y[i] = LAT[i]
					M_l[i,i] = float(1)
				else:
					M_u[i,i] = float(1)

					
			T = np.linalg.inv(M_l + alpha*M_u + beta*L)

			yhatFold = np.matmul(T, y)

			yhatFold = yhatFold[TstIdx]

			# Save the estimated value for this vertex
			for i in range(tstLen):
				# vertex index
				verIdx = TstIdx[i]

				latEst = yhatFold[i]

				yhat[verIdx] = latEst

			fold += 1

		Vec = [abs(yhat[i] - LAT[i]) for i in range(N) if IS_SAMP[i] is True]
		n, bins = np.histogram(Vec, bins=25, range=(0, 250))
		freq = n/sum(n)
		errVecW = []
		for i in range(M):
			elem = Vec[i]
			for j, val in enumerate(bins):
				if val > elem:
					idx = j - 1
					break
			weightedErr = elem*freq[idx]
			errVecW.append(weightedErr)
		errVecW = np.array(errVecW)
		errVec = np.array(Vec)

		mse[a_idx][b_idx] = 1/M*np.sum(errVec ** 2)
		wmse[a_idx][b_idx] = 1/M*np.sum(errVecW ** 2)

		snr[a_idx][b_idx] = 20*np.log10(np.sum(np.array(SAMP_LAT) ** 2)/(M*mse[a_idx][b_idx]))
		wsnr[a_idx][b_idx] = 20*np.log10(np.sum(np.array(SAMP_LAT) ** 2)/(M*wmse[a_idx][b_idx]))

                print('%.2f %.2f %.2f %.2f', mse[a_idx][b_idx], wmse[a_idx][b_idx], snr[a_idx][b_idx], wsnr[a_idx][b_idx])

print()

print('Writing mse to file...')
fid = open(os.path.join('res', 'mse.txt'), 'w')
mse.tofile(fid, sep='\n', format='%.2f')
fid.close()

print('Writing wmse to file...')
fid = open(os.path.join('res','wmse.txt'), 'w')
wmse.tofile(fid, sep='\n', format='%.2f')
fid.close()

print('Writing snr to file...')
fid = open(os.path.join('res','snr.txt'), 'w')
snr.tofile(fid, sep='\n', format='%.2f')
fid.close()

print('Writing wsnr to file...')
fid = open(os.path.join('res', 'wsnr.txt'), 'w')
wsnr.tofile(fid, sep='\n', format='%.2f')
fid.close()

print('\nTest complete.\n')
