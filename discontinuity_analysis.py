
"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""

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


dataDir = 'data/'
meshNames = ['Patient037_I_MESHData9-RV SINUS VOLTAGE.mesh']
latNames = ['Patient037_I_LATSpatialData_9-RV SINUS VOLTAGE_car.txt']

# meshNames = ['Patient034_I_MESHData6-RVFAM SINUS VOLTAGE.mesh']
# latNames = ['Patient034_I_LATSpatialData_6-RVFAM SINUS VOLTAGE_car.txt']

# meshNames = ['Patient033_I_MESHData3-RV FAM PVC A - NORMAL.mesh']
# latNames = ['Patient033_I_LATSpatialData_3-RV FAM PVC A - NORMAL_car.txt']

# meshNames = ['Patient034_I_MESHData5-RVFAM PVC.mesh']
# latNames = ['Patient034_I_LATSpatialData_5-RVFAM PVC_car.txt']

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

S_COORD = [COORD[i] for i in IDX if (COORD[i][0] < 0 and COORD[i][1] < 60 and (float(COORD[i][0]) + 5/4*float(COORD[i][1])) < 20 and COORD[i][2] > 151 and COORD[i][2] < 180)]
S_ORIG_IDX = [i for i in IDX if (COORD[i][0] < 0 and COORD[i][1] < 60 and (float(COORD[i][0]) + 5/4*float(COORD[i][1])) < 20 and COORD[i][2] > 151 and COORD[i][2] < 180)]
S_N = len(S_COORD)
S_IDX = [i for i in range(S_N)]

S_IS_SAMP = [False for i in range(S_N)]
S_LAT = [0 for i in range(S_N)]
for i in range(S_N):
	verIdx = S_IDX[i]
	verOrigIdx = S_ORIG_IDX[i]
	if (verOrigIdx in SAMP_IDX):
		S_IS_SAMP[verIdx] = True
		S_LAT[verIdx] = LAT[verOrigIdx]

S_SAMP_IDX = [S_IDX[i] for i in range(S_N) if S_IS_SAMP[i] is True]
S_SAMP_COORD = [S_COORD[i] for i in range(S_N) if S_IS_SAMP[i] is True]
S_SAMP_LAT = [S_LAT[i] for i in range(S_N) if S_IS_SAMP[i] is True]

S_M = len(S_SAMP_IDX)


S_coordMatrix = np.array(S_COORD)
S_connectMatrix = []

for tri in connectivityMatrix:
	idx0 = int(tri[0])
	idx1 = int(tri[1])
	idx2 = int(tri[2])

	if (idx0 in S_ORIG_IDX) and (idx1 in S_ORIG_IDX) and (idx2 in S_ORIG_IDX):
		s_idx0 = S_IDX[S_ORIG_IDX.index(idx0)]
		s_idx1 = S_IDX[S_ORIG_IDX.index(idx1)]
		s_idx2 = S_IDX[S_ORIG_IDX.index(idx2)]
		S_tri = [s_idx0, s_idx1, s_idx2]
		S_connectMatrix.append(S_tri)

UNKNOWN = [S_COORD[i] for i in range(S_N) if S_IS_SAMP[i] is False]

UNKNOWN = griddata(np.array(S_SAMP_COORD), np.array(S_SAMP_LAT), np.array(UNKNOWN), method='nearest')
currIdx = 0
for i in range(S_N):
	verIdx = S_IDX[i]
	if S_IS_SAMP[i] is False:
		S_LAT[verIdx] = UNKNOWN[currIdx]
		currIdx += 1

minLat = math.floor(min(SAMP_LAT)/10)*10
maxLat = math.ceil(max(SAMP_LAT)/10)*10
elev = 24
azim = -135

pltCoord = np.array(SAMP_COORD)
triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()

# Plot true LAT signal

thisAx = axes[0]
thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
# thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey')
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=SAMP_LAT, cmap='rainbow_r', vmin=minLat, vmax=maxLat, s = 20)

thisAx.set_title('LAT Signal (True)')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01])
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

pltCoord = np.array(S_SAMP_COORD)

triang = mtri.Triangulation(S_coordMatrix[:,0], S_coordMatrix[:,1], triangles=S_connectMatrix)

thisAx = axes[1]
thisAx.plot_trisurf(triang, S_coordMatrix[:,2], color='grey', alpha=0.2)
# thisAx.plot_trisurf(triang, S_coordMatrix[:,2], color=None, edgecolor='k')
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=S_SAMP_LAT, cmap='rainbow_r', vmin=minLat, vmax=maxLat, s = 20)

# for i in range(S_N):
# 	txt = str(S_IDX[i])
# 	thisAx.text(S_coordMatrix[i,0], S_coordMatrix[i,1], S_coordMatrix[i,2], txt)

thisAx.set_title('Section of Interest')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)
# cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height]) # vertical bar to the right
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")
# plt.show()



"""
Get edges and corresponding adjacent triangles.

edges: list of edges, edge = set(v_i, v_j)
triangles: list of corr. triangles adj to each edge, tri = (v_i, v_j, v_k)
"""
print('Generating edge matrix ...')
[EDGES, TRI, excl_midpt] = getE(S_coordMatrix, S_connectMatrix, S_LAT, 50)


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8), subplot_kw=dict(projection="3d"))
pltCoord = np.array(S_SAMP_COORD)

triang = mtri.Triangulation(S_coordMatrix[:,0], S_coordMatrix[:,1], triangles=S_connectMatrix)
ax.plot_trisurf(triang, S_coordMatrix[:,2], color=(0,0,0,0), edgecolor='k', linewidth=.1)
pos = ax.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=S_SAMP_LAT, cmap='rainbow_r', vmin=minLat, vmax=maxLat, s = 20)

# for i in range(S_N):
# 	txt = str(S_IDX[i])
# 	ax.text(S_coordMatrix[i,0], S_coordMatrix[i,1], S_coordMatrix[i,2], txt)

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
A = getUnWeightedAdj(S_coordMatrix, EDGES, TRI)
W = getAdjMatrix(S_coordMatrix, EDGES, TRI)
W1 = getAdjMatrixCotan(S_coordMatrix, EDGES, TRI)

# pltAdjMatrix(W, 0, 20, 'W(i,j) = 1/d(i,j)')
# pltAdjMatrix(W1, 0, 20, 'Cotangent Weights')

D = np.diag(A.sum(axis=1))
I = np.identity(S_N)

print('Calculating Laplacian matrix ...')
L = D - A
# L = D - W
# L = D - W1


""" Hyperparameters """
alpha = 0.001
beta = 1


""" Cross-validation """

folds = 10
kf12 = KFold(n_splits=folds, shuffle=True)

fold = 0

yhat = np.zeros((S_N,1))


minLat = math.floor(min(S_SAMP_LAT)/10)*10
maxLat = math.ceil(max(S_SAMP_LAT)/10)*10

triang = mtri.Triangulation(S_coordMatrix[:,0], S_coordMatrix[:,1], triangles=S_connectMatrix)
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()

plt_idx = 0

# Plot true LAT signal

pltCoord = np.array(S_SAMP_COORD)

for tr_i, tst_i in kf12.split(S_SAMP_IDX):

	# print('\nFold ' + str(fold))

	y = np.zeros((S_N,1))
	M_l = np.zeros((S_N,S_N))
	M_u = np.zeros((S_N,S_N))

	# number of labelled and unlabelled vertices in this fold
	trLen = len(tr_i)
	tstLen = len(tst_i)

	# get vertex indices of labelled/unlabelled nodes
	TrIdx = sorted(np.take(S_SAMP_IDX, tr_i))
	TstIdx = sorted(np.take(S_SAMP_IDX, tst_i))

	# get vertex coordinates
	TrCoord = [S_COORD[i] for i in TrIdx]
	TstCoord = [S_COORD[i] for i in TstIdx]

	# get LAT signal values
	TrVal = [S_LAT[i] for i in TrIdx]
	TstVal = [S_LAT[i] for i in TstIdx]

	# Compute graph interpolation estimate for unlabelled vertices in this fold
	for i in range(S_N):
		if i in TrIdx:
			y[i] = S_LAT[i]
			M_l[i,i] = float(1)
		else:
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
		latTrue = S_LAT[verIdx]

		err = abs(latEst - latTrue)

		# save the estimated value and error for this vertex
		yhat[verIdx] = latEst

		# accumulate squared error
		mse += (err ** 2)
	# average the squared error
	mse = mse/tstLen
	nmse = mse*tstLen/np.sum((np.array(TstVal) - np.mean(TstVal)) ** 2)
	print('Fold ' + str(fold) + ', NMSE:\t' + str(nmse))

	if (fold > 7 and fold < 10):
		pltCoord = np.array(TrCoord)
		pltSig = TrVal

		thisAx = axes[plt_idx]
		thisAx.plot_trisurf(triang, S_coordMatrix[:,2], color='grey', alpha=0.2)
		pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=minLat, vmax=maxLat, s = 20)

		thisAx.set_title('Training (input) signal\nFold '+str(fold))
		thisAx.set_xlabel('X', fontweight ='bold') 
		thisAx.set_ylabel('Y', fontweight ='bold') 
		thisAx.set_zlabel('Z', fontweight ='bold')

		thisAx.view_init(elev, azim)

		plt_idx += 1

		pltCoord = np.array(TstCoord)
		pltSig = TstVal

		thisAx = axes[plt_idx]
		thisAx.plot_trisurf(triang, S_coordMatrix[:,2], color='grey', alpha=0.2)
		pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=minLat, vmax=maxLat, s = 20)

		thisAx.set_title('True values\nFold '+str(fold))
		thisAx.set_xlabel('X', fontweight ='bold') 
		thisAx.set_ylabel('Y', fontweight ='bold') 
		thisAx.set_zlabel('Z', fontweight ='bold')

		thisAx.view_init(elev, azim)

		plt_idx += 1

		pltCoord = np.array(TstCoord)
		pltSig = yhatFold

		thisAx = axes[plt_idx]
		thisAx.plot_trisurf(triang, S_coordMatrix[:,2], color='grey', alpha=0.2)
		# thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey')
		pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=minLat, vmax=maxLat, s = 20)

		thisAx.set_title('Interpolated values\nNMSE = '+str(nmse))
		thisAx.set_xlabel('X', fontweight ='bold') 
		thisAx.set_ylabel('Y', fontweight ='bold') 
		thisAx.set_zlabel('Z', fontweight ='bold')

		thisAx.view_init(elev, azim)

		if (fold == 1):
			cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01])
			plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

		plt_idx += 1

	fold += 1



Vec = [abs(yhat[i] - S_LAT[i]) for i in range(S_N) if S_IS_SAMP[i] is True]
# mxerr = (round(max(Vec)[0]/10)+1)*10
# n, bins = np.histogram(Vec, range=(0, mxerr))
n, bins = np.histogram(Vec, bins=25, range=(0, 250))
# print(bins)
freq = n/sum(n)

errVecW = []
for i in range(S_M):
	elem = Vec[i]
	for j, val in enumerate(bins):
		if val > elem:
			idx = j - 1
			break
	weightedErr = elem*freq[idx]
	errVecW.append(weightedErr)
errVecW = np.array(errVecW)
errVec = np.array(Vec)

sigPower = np.sum((np.array(S_SAMP_LAT) - np.mean(S_SAMP_LAT)) ** 2)

mse = 1/M*np.sum(errVec ** 2)
rmse = np.sqrt(mse)
nmse = np.sum(errVec ** 2)/sigPower
nrmse = rmse/np.mean(S_SAMP_LAT)

snr = 20*np.log10(sigPower/np.sum(errVec ** 2))

print('\n\nMSE:\t{:.2f}'.format(mse))
print('RMSE:\t{:.2f}'.format(rmse))
print('NMSE:\t{:.2f}'.format(nmse))
print('\nSNR:\t{:.2f}'.format(snr))


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()

# Plot true LAT signal

pltCoord = np.array(S_SAMP_COORD)

thisAx = axes[0]
thisAx.plot_trisurf(triang, S_coordMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=S_SAMP_LAT, cmap='rainbow_r', vmin=minLat, vmax=maxLat, s = 20)

thisAx.set_title('LAT Signal (True)')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)
# cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height]) # vertical bar to the right
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

pltSig = yhat[S_SAMP_IDX]

thisAx = axes[1]
thisAx.plot_trisurf(triang, S_coordMatrix[:,2], color='grey', alpha=0.2)
# thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey')
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=minLat, vmax=maxLat, s = 20)

thisAx.set_title('LAT Signal (Graph Estimation)')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
thisAx.view_init(elev, azim)
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01])
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")


plt.show()