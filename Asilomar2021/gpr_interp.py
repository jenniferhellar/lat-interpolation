"""
--------------------------------------------------------------------------------
Gaussian Process Regression (GPR) interpolation on LAT data.
--------------------------------------------------------------------------------

Description: 10-fold cross-validation to randomly select 10 test
sets for interpolation.  5x repetitition for error mean and
variance estimation.

Requirements: numpy, matplotlib, sklearn, scipy

File: gpr_interp.py

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""


import numpy as np

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

# Gaussian process regression interpolation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


COMPUTE_WEIGHTED_ERROR		=		0
PLOT_FINAL_EST				=		0


""" Read the files """
dataDir = 'data/'
meshFile = 'MESHData.mesh'
latFile = 'LATSpatialData.txt'
nm = meshFile[0:-5]

print('Reading files for ' + nm + ' ...')
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[latCoords, latVals] = readLAT(dataDir + latFile)

N = len(coordinateMatrix)	# number of vertices in the graph
M = len(latCoords)			# number of raw signal samples

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
	LAT[verIdx] = latVals[i]	# assign known LAT values

# (short) lists of sampled vertices, coordinates, and  LAT values
SAMP_IDX = [IDX[i] for i in range(N) if IS_SAMP[i] is True]
SAMP_COORD = [COORD[i] for i in range(N) if IS_SAMP[i] is True]
SAMP_LAT = [LAT[i] for i in range(N) if IS_SAMP[i] is True]

M = len(SAMP_IDX)	# number of signal samples


""" Create GPR kernel and regressor """
gp_kernel = RBF(length_scale=0.01) + RBF(length_scale=0.1) + RBF(length_scale=1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)


""" Cross-validation """
folds = 10
# kf = KFold(n_splits=folds, shuffle=True)
rkf = RepeatedKFold(n_splits=folds, n_repeats=5, random_state=1)

fold = 0
yhatGPR = np.zeros((N,1))

# for tr_i, tst_i in kf.split(SAMP_IDX):
for tr_i, tst_i in rkf.split(SAMP_IDX):
	# on each repeat, compute the entire map again
	if (fold % folds) == 0:
		yhatGPR = np.zeros((N,1))

	# number of labelled and unlabelled vertices
	trLen = len(tr_i)
	tstLen = len(tst_i)

	# get global vertex indices of labelled/unlabelled nodes
	TrIdx = sorted(np.take(SAMP_IDX, tr_i))
	TstIdx = sorted(np.take(SAMP_IDX, tst_i))

	# get vertex coordinates
	TrCoord = [COORD[i] for i in TrIdx]
	TstCoord = [COORD[i] for i in TstIdx]

	# get LAT signal values
	TrVal = [LAT[i] for i in TrIdx]
	TstVal = [LAT[i] for i in TstIdx]

	# Fit the GPR with training samples
	gpr.fit(TrCoord, TrVal)

	# Predict the test LAT values
	yhatGPRfold = gpr.predict(TstCoord, return_std=False)

	# Calculate the mean squared error for this fold
	mseGPR = 0
	for i in range(tstLen):
		verIdx = TstIdx[i]		# global vertex index

		latEst = yhatGPRfold[i]
		latTrue = LAT[verIdx]

		err = abs(latEst - latTrue)

		# save the estimated value for this vertex
		yhatGPR[verIdx] = latEst

		# accumulate squared error
		mseGPR += (err ** 2)
	# average and normalize the squared error
	mseGPR = mseGPR/tstLen
	nmse = float(mseGPR*tstLen/np.sum((np.array(TstVal) - np.mean(TstVal)) ** 2))
	print('Fold {:g}, NMSE:\t{:.2f}'.format(fold % folds, nmse))

	fold += 1

	# if one repetition done, compute overall NMSE and SNR
	if (fold % folds) == 0:
		# |error| for each sampled vertex
		Vec = [abs(yhatGPR[i] - LAT[i]) for i in range(N) if IS_SAMP[i] is True]
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
		print('SNR:\t{:.4f}\n\n'.format(snr))


if COMPUTE_WEIGHTED_ERROR:
	Vec = [abs(yhatGPR[i] - LAT[i]) for i in range(N) if IS_SAMP[i] is True]
	n, bins = np.histogram(Vec, bins=25, range=(0, 250))
	# print(bins)
	freq = n/sum(n)

	errVecGPR = []
	for i in range(M):
		elem = Vec[i]
		for j, val in enumerate(bins):
			if val > elem:
				idx = j - 1
				break
		weightedErr = elem*freq[idx]
		errVecGPR.append(weightedErr)
	errVecWGPR = np.array(errVecGPR)

	sigPower = np.sum((np.array(SAMP_LAT) - np.mean(SAMP_LAT)) ** 2)
	wmse = 1/M*np.sum(errVecWGPR ** 2)
	wsnr = 20*np.log10(sigPower/np.sum(errVecWGPR ** 2))

	print('\n\nWMSE:\t{:.2f}'.format(wmse))
	print('WSNR:\t{:.2f}'.format(wsnr))

if PLOT_FINAL_EST:
	pltCoord = np.array(SAMP_COORD)
	
	triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

	fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(16, 8), subplot_kw=dict(projection="3d"))
	axes = ax.flatten()

	# Plot true LAT signal
	thisAx = axes[0]
	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=SAMP_LAT, cmap='rainbow_r', vmin=-200, vmax=50, s = 20)

	thisAx.set_title('LAT Signal (True)')
	thisAx.set_xlabel('X', fontweight ='bold') 
	thisAx.set_ylabel('Y', fontweight ='bold') 
	thisAx.set_zlabel('Z', fontweight ='bold')
	cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01])
	plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

	# Plot overall estimated LAT signal (aggregated from computation in each fold)
	thisAx = axes[1]
	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=yhatGPR[SAMP_IDX], cmap='rainbow_r', vmin=-200, vmax=50, s = 20)

	thisAx.set_title('LAT Signal (Estimated)')
	thisAx.set_xlabel('X', fontweight ='bold') 
	thisAx.set_ylabel('Y', fontweight ='bold') 
	thisAx.set_zlabel('Z', fontweight ='bold')
	# cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height]) # vertical bar to the right
	cax = fig.add_axes([thisAx.get_position().x0+0.015,thisAx.get_position().y0-0.05,thisAx.get_position().width,0.01]) # horiz bar on the bottom
	plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")
	plt.show()