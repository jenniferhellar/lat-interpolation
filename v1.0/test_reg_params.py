"""
--------------------------------------------------------------------------------
Test MAGIC-LAT over different regularization parameters.
--------------------------------------------------------------------------------

Description: Computes MAGIC-LAT error metrics for test sets of 87 vertices with
either coarse or fine regularization parameters.  Results written out to text
files.

Results independently plotted in 
	reg_params_coarse.png
	reg_params_fine.png.

Requirements: os, numpy, sklearn, scipy, math

File: test_reg_params.py

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""

import os

import numpy as np
import math

# KD-Tree for mapping to nearest point
from scipy.spatial import cKDTree

# cross-validation package
from sklearn.model_selection import KFold

# nearest-neighbor interpolation
from scipy.interpolate import griddata

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT

# functions for computing graph edges and adj matrix
from utils import *


COARSE 		=		1
SPARSIFY_THRESHOLD			=		50

""" Hyperparameters """
if COARSE:
	alphas = [0.01, 0.1, 1, 5, 10, 100, 1000]
	betas = [0.01, 0.1, 1, 5, 10, 100, 1000]
else:
	alphas = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
	betas = [1.0, 2.0, 4.0, 5.0, 8.0, 10.0]

# create a results directory
if not os.path.isdir('res'):
	os.makedirs('res')

# write out the alphas and betas chosen
print('Writing alphas to file...')
fid = open(os.path.join('res','alphas.txt'), 'w')
np.array(alphas).tofile(fid, sep='\n', format='%.2f')
fid.close()

print('Writing betas to file...')
fid = open(os.path.join('res','betas.txt'), 'w')
np.array(betas).tofile(fid, sep='\n', format='%.2f')
fid.close()


""" Read the files """
dataDir = 'data/'
meshFile = 'MESHData.mesh'
latFile = 'LATSpatialData.txt'
nm = meshFile[0:-5]

print('Reading files for ' + nm + ' ...\n')
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[latCoords, latVals] = readLAT(dataDir + latFile)

N = len(coordinateMatrix)	# number of vertices in the graph
M = len(latCoords)			# number of signal samples

IDX = [i for i in range(N)]		# ordered vertex indices and coordinates
COORD = [coordinateMatrix[i] for i in IDX]

# Map data points to mesh coordinates
coordKDtree = cKDTree(coordinateMatrix)
[dist, idxs] = coordKDtree.query(latCoords, k=1)

IS_SAMP = [False for i in range(N)]		# binary indicator for vertex sampled or not
LAT = [0 for i in range(N)]
for i in range(M):
	verIdx = idxs[i]
	IS_SAMP[verIdx] = True
	LAT[verIdx] = latVals[i]

# (short) lists of sampled vertices, coordinates, and  LAT values
SAMP_IDX = [IDX[i] for i in range(N) if IS_SAMP[i] is True]
SAMP_COORD = [COORD[i] for i in range(N) if IS_SAMP[i] is True]
SAMP_LAT = [LAT[i] for i in range(N) if IS_SAMP[i] is True]

M = len(SAMP_IDX)

# NN interpolation of unknown vertices
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

"""
Get edges and corresponding adjacent triangles.

EDGES: list of edges, edge = set(v_i, v_j)
excl_midpt: midpoint of each removed edge (for visualization)
"""

print('Generating edge matrix ...')
[EDGES, excl_midpt] = getE(coordinateMatrix, connectivityMatrix, LAT, SPARSIFY_THRESHOLD)


""" 
A: NxN unweighted adjacency matrix
D: NxN degree matrix
I: NxN identity matrix 

L: NxN Laplace matrix

"""
print('Calculating adjacency matrix ...')
A = getUnWeightedAdj(N, EDGES)
D = np.diag(A.sum(axis=1))
I = np.identity(N)

print('Calculating Laplacian matrix ...')
L = D - A


""" Cross-validation """
folds = 10
kf12 = KFold(n_splits=folds, shuffle=True)

mse = np.zeros((len(alphas), len(betas)))
nmse = np.zeros((len(alphas), len(betas)))
rmse = np.zeros((len(alphas), len(betas)))
wmse = np.zeros((len(alphas), len(betas)))
snr = np.zeros((len(alphas), len(betas)))

for a_idx in range(len(alphas)):
	alpha = alphas[a_idx]

	for b_idx in range(len(betas)):
		beta = betas[b_idx]

		print('\nCross-validating alpha=' + str(alpha) + ', beta=' + str(beta))

		fold = 0

		yhat = np.zeros((N,1))

		for tr_i, tst_i in kf12.split(SAMP_IDX):

			print('\tFold ' + str(fold))

			# number of labelled and unlabelled vertices in this fold
			trLen = len(tr_i)
			tstLen = len(tst_i)

			# get vertex indices of labelled/unlabelled nodes
			TrIdx = sorted(np.take(SAMP_IDX, tr_i))
			TstIdx = sorted(np.take(SAMP_IDX, tst_i))

			# partially sampled input y and masking matrices M_l and M_u
			y = np.zeros((N,1))			
			M_l = np.zeros((N,N))
			M_u = np.zeros((N,N))

			for i in range(N):
				if i in TrIdx:
					y[i] = LAT[i]
					M_l[i,i] = float(1)
				else:
					M_u[i,i] = float(1)

					
			T = np.linalg.inv(M_l + alpha*M_u + beta*L)

			# Compute graph interpolation estimate for unlabelled vertices in this fold
			yhatFold = np.matmul(T, y)
			yhatFold = yhatFold[TstIdx]

			# Save the estimated value for this vertex
			for i in range(tstLen):
				verIdx = TstIdx[i]
				latEst = yhatFold[i]
				yhat[verIdx] = latEst

			fold += 1

		# when entire estimate done (10 folds), compute error metrics
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

		sigPower = np.sum((np.array(SAMP_LAT) - np.mean(SAMP_LAT)) ** 2)

		mse[a_idx][b_idx] = 1/M*np.sum(errVec ** 2)
		nmse[a_idx][b_idx] = np.sum(errVec ** 2)/sigPower
		rmse[a_idx][b_idx] = np.sqrt(mse[a_idx][b_idx])
		wmse[a_idx][b_idx] = 1/M*np.sum(errVecW ** 2)

		snr[a_idx][b_idx] = 20*np.log10(sigPower/(M*mse[a_idx][b_idx]))

		print('{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(mse[a_idx][b_idx], nmse[a_idx][b_idx], rmse[a_idx][b_idx], wmse[a_idx][b_idx], snr[a_idx][b_idx]))

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

print('Writing nmse to file...')
fid = open(os.path.join('res', 'nmse.txt'), 'w')
nmse.tofile(fid, sep='\n', format='%.2f')
fid.close()

print('Writing rmse to file...')
fid = open(os.path.join('res', 'rmse.txt'), 'w')
rmse.tofile(fid, sep='\n', format='%.2f')
fid.close()

print('\nTest complete.\n')
