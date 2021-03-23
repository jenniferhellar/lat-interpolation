
"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""

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



dataDir = 'data/'
meshNames = ['Patient037_I_MESHData9-RV SINUS VOLTAGE.mesh']
latNames = ['Patient037_I_LATSpatialData_9-RV SINUS VOLTAGE_car.txt']

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


""" Cross-validation """

folds = 10
kf12 = KFold(n_splits=folds, shuffle=True)

fold = 0
errVecNN = np.zeros((N,1))
errVec = np.zeros((N,1))
yhatNN = np.zeros((N,1))

for tr_i, tst_i in kf12.split(SAMP_IDX):
	# number of labelled and unlabelled vertices in this fold
	trLen = len(tr_i)
	tstLen = len(tst_i)

	print('\nFold ' + str(fold) + '\t# of labelled vertices: ' + str(trLen) + '\t# of unlabelled vertices: ' + str(tstLen))

	# get vertex indices of labelled/unlabelled nodes
	TrIdx = sorted(np.take(SAMP_IDX, tr_i))
	TstIdx = sorted(np.take(SAMP_IDX, tst_i))

	# get vertex coordinates
	TrCoord = [COORD[i] for i in TrIdx]
	TstCoord = [COORD[i] for i in TstIdx]

	# get LAT signal values
	TrVal = [LAT[i] for i in TrIdx]
	TstVal = [LAT[i] for i in TstIdx]

	# Compute nearest neighbor estimate for unlabelled vertices in this fold
	yhatNNfold = griddata(np.array(TrCoord), np.array(TrVal), np.array(TstCoord), method='nearest')

	# Calculate the mean squared error for this fold
	mseNN = 0
	for i in range(tstLen):
		# vertex index
		verIdx = TstIdx[i]

		latEst = yhatNNfold[i]
		latTrue = LAT[verIdx]

		# calculate error
		err = latEst - latTrue

		# save the estimated value and error for this vertex
		yhatNN[verIdx] = latEst
		errVecNN[verIdx] = err

		# accumulate squared error
		mseNN += (err ** 2)
	# average the squared error
	mseNN = mseNN/tstLen
	print('NN-Estimate MSE:\t' + str(mseNN))

	fold += 1

errVecNN = errVecNN[SAMP_IDX]

print(np.sum(abs(errVecNN) < 15)/M)
print(np.sum(abs(errVecNN) < 10)/M)
print(np.sum(abs(errVecNN) < 5)/M)

x = [i for i in range(M)]

# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16,8))
# ax.scatter(x, errVecNN)
# ax.set_title('NN Error')
# plt.xlabel('Vertex')
# plt.ylabel('Estimation Error')
# plt.show()



pltCoord = np.array(SAMP_COORD)
triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()

# Plot true LAT signal
thisAx = axes[0]
thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=SAMP_LAT, cmap='rainbow_r', s = 20)

thisAx.set_title('LAT Signal (True)')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01])
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

# Plot overall estimated LAT signal (aggregated from computation in each fold)
thisAx = axes[1]
thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=yhatNN[SAMP_IDX], cmap='rainbow_r', s = 20)

thisAx.set_title('LAT Signal (Estimated)')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')
# cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height]) # vertical bar to the right
cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")
plt.show()