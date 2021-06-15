"""
--------------------------------------------------------------------------------
Manifold Approximating Graph Interpolation on Cardiac LAT data (MAGIC-LAT).
--------------------------------------------------------------------------------

Description: Cross-validation to randomly select test sets for interpolation.  
5x repetitition for error mean and variance estimation.

Requirements: os, numpy, matplotlib, sklearn, scipy, math

File: gpr_interp.py

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
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
from sklearn.model_selection import RepeatedKFold

# nearest-neighbor interpolation
from scipy.interpolate import griddata

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


from utils import *
from const import *

SPARSIFY_THRESHOLD			=		50
PLOT_EDGE_MOD				=		0
COMPUTE_WEIGHTED_ERROR		=		0
PLOT_FINAL_EST				=		0

PLOT_EDGE_MOD		=		0
PLOT_FOLD			=		0

# 20 is the one I have been working with
mapIdx = 20

""" Read the files """
meshFile = meshNames[mapIdx]
latFile = latNames[mapIdx]
nm = meshFile[0:-5]

print('Reading files for ' + nm + ' ...\n')
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[latCoords, latVals] = readLAT(dataDir + latFile)

N = len(coordinateMatrix)	# number of vertices in the graph
M = len(latCoords)			# number of signal samples

if N > 11000:
	print('Graph too large!')
	exit(0)

IDX = [i for i in range(N)]		# binary indicator for vertex sampled or not
COORD = [coordinateMatrix[i] for i in IDX]

# Map data points to mesh coordinates
coordKDtree = cKDTree(coordinateMatrix)
[dist, idxs] = coordKDtree.query(latCoords, k=1)

# (short) lists of sampled vertices, coordinates, and  LAT values
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

# Figure view
elev = 24
azim = -135

# elev = 70
# azim = 45

# azim = -180
# elev = -75


"""
Get edges and corresponding adjacent triangles.

edges: list of edges, edge = set(v_i, v_j)
triangles: list of corr. triangles adj to each edge, tri = (v_i, v_j, v_k)
"""

print('Generating edge matrix ...')
threshold = 50
[EDGES, TRI, excl_midpt] = getE(coordinateMatrix, connectivityMatrix, LAT, threshold)
print('Edges excluded: ', str(len(excl_midpt)))

if PLOT_EDGE_MOD:
	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8), subplot_kw=dict(projection="3d"))
	pltCoord = np.array(SAMP_COORD)

	triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)
	ax.plot_trisurf(triang, coordinateMatrix[:,2], color=(0,0,0,0), edgecolor='k', linewidth=.1)
	pos = ax.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=SAMP_LAT, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 20)

	# for i in range(S_N):
	# 	txt = str(S_IDX[i])
	# 	ax.text(coordinateMatrix[i,0], coordinateMatrix[i,1], coordinateMatrix[i,2], txt)

	for i in range(len(excl_midpt)):
		txt = 'x'
		ax.text(excl_midpt[i,0], excl_midpt[i,1], excl_midpt[i,2], txt, color='k', fontsize='x-small')

	ax.set_title('Section of Interest')
	ax.set_xlabel('X', fontweight ='bold') 
	ax.set_ylabel('Y', fontweight ='bold') 
	ax.set_zlabel('Z', fontweight ='bold')
	ax.view_init(elev, azim)
	# cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height]) # vertical bar to the right
	cax = fig.add_axes([ax.get_position().x0+0.015,ax.get_position().y0-0.05,ax.get_position().width,0.01]) # horiz bar on the bottom
	plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")
	# plt.show()
	# exit()


""" 
A: NxN unweighted adjacency matrix
W: NxN weighted adjacency matrix (cotan weights)
D: NxN degree matrix
I: NxN identity matrix 

L: NxN Laplace matrix

"""
print('Calculating adjacency matrices ...')
A = getUnWeightedAdj(N, EDGES)
# W = getAdjMatrix(coordinateMatrix, EDGES, TRI)
# W = getAdjMatrixCotan(coordinateMatrix, EDGES, TRI)
# W = getAdjMatrixExp(coordinateMatrix, EDGES, TRI)

# pltAdjMatrix(W, 0, 20, 'W(i,j) = 1/d(i,j)')
# pltAdjMatrix(W1, 0, 20, 'Cotangent Weights')

D = np.diag(A.sum(axis=1))
I = np.identity(N)

print('Calculating Laplacian matrix ...')
L = D - A
# L = D - W

# lambdas, U = np.linalg.eigh(L)

# print('Writing lambdas to file...')
# fid = open('V.npy', 'wb')
# np.save(fid, lambdas)
# fid.close()

# print('Writing U to file...')
# fid = open('U.npy', 'wb')
# np.save(fid, U)
# fid.close()

# lambdas = np.load('V.npy')
# U = np.load('U.npy')
# Ut = np.transpose(U)

# V = np.diag(lambdas)


""" Hyperparameters found by cross-validation """
alpha = 1e-03
beta = 1

# S = np.linalg.inv(I + alpha*V)

# tmp = np.matmul(S, Ut)
# T = np.matmul(U, tmp)


""" 
Cross-validation w/5x repetition

Varied interpolation levels:

SWITCH_TST_TR	folds	numTrain	numTest
0				10		780			87
0				8		758			109
0				6		722			145
0				4		650			217
0				3		578			289
0				2		433			434

1				3		289			578
1				4		217			650
1				6		145			722
1				8		109			758
1				10		87			780

"""

SWITCH_TST_TR		=		0

folds = 10
# kf = KFold(n_splits=folds, shuffle=True)
rkf = RepeatedKFold(n_splits=folds, n_repeats=5, random_state=1)

fold = 0
yhat = np.zeros((N,1))

if PLOT_FOLD:
	triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)
	fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(16, 8), subplot_kw=dict(projection="3d"))
	axes = ax.flatten()

	plt_idx = 0

	# Plot true LAT signal

	pltCoord = np.array(SAMP_COORD)

first = True

# for tr_i, tst_i in kf.split(SAMP_IDX):
for tr_i, tst_i in rkf.split(SAMP_IDX):

	if (fold % folds) == 0:
		yhat = np.zeros((N,1))

	if (SWITCH_TST_TR):
		tmp = tr_i
		tr_i = tst_i
		tst_i = tmp

	if first:
		# print('{:.2f}% interpolation'.format(len(tst_i)/M*100))
		# print('{:2f}%'.format(len(tr_i)/N*100))
		print(len(tr_i))
		print(len(tst_i))
		first = False

	y = np.zeros((N,1))
	M_l = np.zeros((N,N))
	M_u = np.zeros((N,N))

	# number of labelled and unlabelled vertices in this fold
	trLen = len(tr_i)
	tstLen = len(tst_i)

	# get vertex indices of labelled/unlabelled nodes
	TrIdx = sorted(np.take(SAMP_IDX, tr_i))
	TstIdx = sorted(np.take(SAMP_IDX, tst_i))

	# get vertex coordinates
	TrCoord = [COORD[i] for i in TrIdx]
	TstCoord = [COORD[i] for i in TstIdx]

	# get LAT signal values
	TrVal = [LAT[i] for i in TrIdx]
	TstVal = [LAT[i] for i in TstIdx]

	# Compute graph interpolation estimate for unlabelled vertices in this fold
	# TrCoordKDtree = cKDTree(TrCoord)
	for i in range(N):
		if i in TrIdx:
			y[i] = LAT[i]
			M_l[i,i] = float(1)
		else:
			# [dist, idxs] = TrCoordKDtree.query(COORD[i], k=1)
			# y[i] = TrVal[idxs]
			M_u[i,i] = float(1)

			
	T = np.linalg.inv(M_l + alpha*M_u + beta*L)

	yhatFold = np.matmul(T, y)

	yhatFold = yhatFold[TstIdx]

	# Calculate the mean squared error for this fold
	mse = 0
	for i in range(tstLen):
		# vertex index
		verIdx = TstIdx[i]

		latEst = yhatFold[i]
		latTrue = LAT[verIdx]

		err = abs(latEst - latTrue)

		# save the estimated value and error for this vertex
		yhat[verIdx] = latEst

		# accumulate squared error
		mse += (err ** 2)
	# average the squared error
	# mse = mse/tstLen
	# nmse = float(mse*tstLen/np.sum((np.array(TstVal) - np.mean(TstVal)) ** 2))
	# print('Fold {:g}, NMSE:\t{:.2f}'.format(fold, nmse))

	if (fold == 0) and PLOT_FOLD:

		pltCoord = np.array(TrCoord)
		pltSig = TrVal

		thisAx = axes[plt_idx]
		thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
		pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 20)

		thisAx.set_title('Training (input) signal\nFold '+str(fold))
		thisAx.set_xlabel('X', fontweight ='bold') 
		thisAx.set_ylabel('Y', fontweight ='bold') 
		thisAx.set_zlabel('Z', fontweight ='bold')

		thisAx.view_init(elev, azim)

		plt_idx += 1

		pltCoord = np.array(TstCoord)
		pltSig = TstVal

		thisAx = axes[plt_idx]
		thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
		pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 20)

		thisAx.set_title('True values\nFold '+str(fold))
		thisAx.set_xlabel('X', fontweight ='bold') 
		thisAx.set_ylabel('Y', fontweight ='bold') 
		thisAx.set_zlabel('Z', fontweight ='bold')

		thisAx.view_init(elev, azim)

		plt_idx += 1

		pltCoord = np.array(TstCoord)
		pltSig = yhatFold

		thisAx = axes[plt_idx]
		thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
		# thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey')
		pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 20)

		thisAx.set_title('Interpolated values\nNMSE = {:.2f}'.format(nmse))
		thisAx.set_xlabel('X', fontweight ='bold') 
		thisAx.set_ylabel('Y', fontweight ='bold') 
		thisAx.set_zlabel('Z', fontweight ='bold')

		thisAx.view_init(elev, azim)

		if (fold == 1):
			cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01])
			plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

		plt_idx += 1

	fold += 1

	if (fold % folds) == 0:
		Vec = [abs(yhat[i] - LAT[i]) for i in range(N) if IS_SAMP[i] is True]
		errVec = np.array(Vec)
		sigPower = np.sum((np.array(SAMP_LAT) - np.mean(SAMP_LAT)) ** 2)

		mse = 1/M*np.sum(errVec ** 2)
		rmse = np.sqrt(mse)
		nmse = np.sum(errVec ** 2)/sigPower
		nrmse = rmse/np.mean(SAMP_LAT)

		snr = 20*np.log10(sigPower/np.sum(errVec ** 2))

		# print('\n\nMSE:\t{:.4f}'.format(mse))
		# print('RMSE:\t{:.4f}'.format(rmse))
		print('NMSE:\t{:.4f}'.format(nmse))
		print('SNR:\t{:.4f}\n'.format(snr))

exit(0)
Vec = [abs(yhat[i] - LAT[i]) for i in range(N) if IS_SAMP[i] is True]
# mxerr = (round(max(Vec)[0]/10)+1)*10
# n, bins = np.histogram(Vec, range=(0, mxerr))
n, bins = np.histogram(Vec, bins=25, range=(0, 250))
# print(bins)
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

mse = 1/M*np.sum(errVec ** 2)
rmse = np.sqrt(mse)
nmse = np.sum(errVec ** 2)/sigPower
nrmse = rmse/np.mean(SAMP_LAT)

snr = 20*np.log10(sigPower/np.sum(errVec ** 2))

print('\n\nMSE:\t{:.4f}'.format(mse))
print('RMSE:\t{:.4f}'.format(rmse))
print('NMSE:\t{:.4f}'.format(nmse))
print('\nSNR:\t{:.4f}'.format(snr))


wmse = 1/M*np.sum(errVecW ** 2)

wsnr = 20*np.log10(sigPower/np.sum(errVecW ** 2))

print('\n\nWMSE:\t{:.2f}'.format(wmse))
print('WSNR:\t{:.2f}'.format(wsnr))


if PLOT_ERROR_VEC:
	x = [i for i in range(M)]

	fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(16,8))
	axes = ax.flatten()

	thisAx = axes[0]
	thisAx.scatter(x, errVec)
	thisAx.set_title('|Error|')
	thisAx.set_xlabel('Vertex')
	thisAx.set_ylabel('Error (ms)')

	thisAx = axes[1]
	thisAx.scatter(x, errVecW)
	thisAx.set_title('|Error|*(frequency of error)')
	thisAx.set_xlabel('Vertex')
	thisAx.set_ylabel('Error Weighted by Occurrence Frequency\nbin width='+str(250/(len(bins)-1))+'ms')
		
	# plt.show()

# n, bins, patches = plt.hist(x=errVec, bins='auto', color='#0504aa', rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Error')
# plt.ylabel('Frequency')
# plt.title('Graph Estimation Error Histogram')
# # plt.text(23, 45, r'$\alpha=0.1, \beta=1$')
# maxfreq = n.max()
# # Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# plt.show()


if PLOT_OUTPUT:
	pltCoord = np.array(SAMP_COORD)
	triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

	fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8), subplot_kw=dict(projection="3d"))
	axes = ax.flatten()

	# Plot true LAT signal
	thisAx = axes[0]
	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	# thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey')
	pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=SAMP_LAT, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 20)

	thisAx.set_title('LAT Signal (True)')
	thisAx.set_xlabel('X', fontweight ='bold') 
	thisAx.set_ylabel('Y', fontweight ='bold') 
	thisAx.set_zlabel('Z', fontweight ='bold')
	thisAx.view_init(elev, azim)
	cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01])
	plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

	# Plot overall estimated LAT signal (aggregated from computation in each fold)
	pltSig = yhat[SAMP_IDX]
	pltCoord = np.array(SAMP_COORD)

	thisAx = axes[1]
	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 20)

	thisAx.set_title('LAT Signal (Graph Estimation)')
	thisAx.set_xlabel('X', fontweight ='bold') 
	thisAx.set_ylabel('Y', fontweight ='bold') 
	thisAx.set_zlabel('Z', fontweight ='bold')
	thisAx.view_init(elev, azim)
	# cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height]) # vertical bar to the right
	cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
	plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")
	plt.show()