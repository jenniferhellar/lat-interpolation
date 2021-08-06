"""
--------------------------------------------------------------------------------
Utility functions for MAGIC-LAT.
--------------------------------------------------------------------------------

Description: Utility functions to compute graph edges from triangle mesh and
corresponding adjacency matrix from the edges.

Requirements: numpy

File: utils.py

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import norm
import math

import os

import cv2
from vedo import *

import matplotlib.cm as cm
from matplotlib.colors import Normalize
import colour

# KD-Tree for mapping to nearest point
from scipy.spatial import cKDTree


def findSpatioMeanAndSigma(bins, binEdges, fig, nonEmpty):

	dimX = fig.shape[0]
	dimY = fig.shape[1]

	mu = np.zeros((bins, bins, bins, 2))
	sigma = np.zeros((bins, bins, bins, 2, 2))

	for r_i, r_v in enumerate(binEdges):
		for g_i, g_v in enumerate(binEdges):
			for b_i, b_v in enumerate(binEdges):
				if r_v < 256 and g_v < 256 and b_v < 256:
					if nonEmpty[r_i, g_i, b_i]:
						pxls = np.column_stack(np.where((fig[:, :, 0] >= r_v) & (fig[:, :, 0] < binEdges[r_i + 1]) & \
							(fig[:, :, 1] >= g_v) & (fig[:, :, 1] < binEdges[g_i + 1]) & \
							(fig[:, :, 2] >= b_v) & (fig[:, :, 2] < binEdges[b_i + 1])))
						if pxls.shape[0] == 0:
							mu[r_i, g_i, b_i] = np.array([[dimX/2, dimY/2]])
							sigma[r_i, g_i, b_i] = np.array([[1, 0], [0, 1]])
						else:
							mu[r_i, g_i, b_i] = np.mean(pxls, axis=0)
							sigma[r_i, g_i, b_i] = np.cov(pxls, rowvar=0)
	return mu, sigma


def js(mu1, sigma1, n1, mu2, sigma2, n2):
	if (n1 == 0 and n2 == 0):
		return 0

	mu_hat = (n2*mu1 + n1*mu2)/(n1 + n2)

	sigma_hat = n2/(n1+n2) * (sigma1 + np.matmul(mu1, mu1.T)) + \
		n1/(n1+n2) * (sigma2 + np.matmul(mu2, mu2.T)) - np.matmul(mu_hat, mu_hat.T)

	if np.linalg.det(sigma1) <= 0 or np.linalg.det(sigma2) <= 0 or np.linalg.det(sigma_hat) <= 0:
		return 0

	t0 = math.log(np.linalg.det(sigma_hat))

	t1 = 1/2 * np.trace(np.matmul(np.linalg.inv(sigma_hat), (sigma1 + sigma2)))

	t2 = 1/4 * np.matmul((mu1 - mu2).T, np.matmul(np.linalg.inv(sigma_hat), (mu1 - mu2)))

	t3 = 1/2 * math.log(np.linalg.det(sigma1) * np.linalg.det(sigma2))

	js_div = t0 + t1 - 2 + t2 - t3

	return js_div


def spatioCorr(bins, nonEmpty, trueMean, trueSigma, trueHist, 
	estMean, estSigma, estHist):
	spatcorr = 0

	for i in range(bins):
		for j in range(bins):
			for k in range(bins):
				if nonEmpty[i,j,k]:
					estjs = js(trueMean[i,j,k], trueSigma[i,j,k], trueHist[i,j,k], 
						estMean[i,j,k], estSigma[i,j,k], estHist[i,j,k])

					# Histogram correlation
					num = trueHist[i,j,k] * estHist[i,j,k]
					denom = math.sqrt(np.sum(trueHist ** 2) * np.sum(estHist ** 2))
					spatcorr += (num / denom) * math.exp(-1 * estjs)

					# # Histogram intersection
					# spatcorr += min(trueHist[i, j, k], estHist[i, j, k])*math.exp(-1 * estjs)

	return spatcorr

def colorHistAndSpatioCorr(outDir, trueImageFile,
		magicImageFile, gprImageFile, quLATiImageFile,
		bins, binEdges, outputFileSuffix):
	img = cv2.imread(os.path.join(outDir, trueImageFile), cv2.IMREAD_GRAYSCALE)
	n_black_px = np.sum(img == 0)
	# numpx = np.sum(img > 0)

	figTruth = cv2.imread(os.path.join(outDir, trueImageFile))
	figTruth = cv2.cvtColor(figTruth, cv2.COLOR_BGR2RGB)
	
	true_hist = cv2.calcHist([figTruth], [0, 1, 2], None, [bins, bins, bins],
		[0, 256, 0, 256, 0, 256])
	true_hist[0, 0, 0] -= n_black_px
	cv2.normalize(true_hist, true_hist)
	true_hist_flat = true_hist.flatten()

	figEst = cv2.imread(os.path.join(outDir, magicImageFile))
	figEst = cv2.cvtColor(figEst, cv2.COLOR_BGR2RGB)
	
	magic_hist = cv2.calcHist([figEst], [0, 1, 2], None, [bins, bins, bins],
		[0, 256, 0, 256, 0, 256])
	magic_hist[0, 0, 0] -= n_black_px
	cv2.normalize(magic_hist, magic_hist)
	magic_hist_flat = magic_hist.flatten()

	figEstGPR = cv2.imread(os.path.join(outDir, gprImageFile))
	figEstGPR = cv2.cvtColor(figEstGPR, cv2.COLOR_BGR2RGB)
	
	gpr_hist = cv2.calcHist([figEstGPR], [0, 1, 2], None, [bins, bins, bins],
		[0, 256, 0, 256, 0, 256])
	gpr_hist[0, 0, 0] -= n_black_px
	cv2.normalize(gpr_hist, gpr_hist)
	gpr_hist_flat = gpr_hist.flatten()

	figEstquLATi = cv2.imread(os.path.join(outDir, quLATiImageFile))
	figEstquLATi = cv2.cvtColor(figEstquLATi, cv2.COLOR_BGR2RGB)
	
	quLATi_hist = cv2.calcHist([figEstquLATi], [0, 1, 2], None, [bins, bins, bins],
		[0, 256, 0, 256, 0, 256])
	quLATi_hist[0, 0, 0] -= n_black_px
	cv2.normalize(quLATi_hist, quLATi_hist)
	quLATi_hist_flat = quLATi_hist.flatten()

	nonEmpty = np.zeros((bins, bins, bins))
	for r_i, r_v in enumerate(binEdges):
		for g_i, g_v in enumerate(binEdges):
			for b_i, b_v in enumerate(binEdges):
				if r_v < 256 and g_v < 256 and b_v < 256:
					if true_hist[r_i, g_i, b_i] > 0 or magic_hist[r_i, g_i, b_i] > 0 or gpr_hist[r_i, g_i, b_i] > 0:
						nonEmpty[r_i, g_i, b_i] = True

	true_mean, true_sigma = findSpatioMeanAndSigma(bins, binEdges, figTruth, nonEmpty)
	magic_mean, magic_sigma = findSpatioMeanAndSigma(bins, binEdges, figEst, nonEmpty)
	gpr_mean, gpr_sigma = findSpatioMeanAndSigma(bins, binEdges, figEstGPR, nonEmpty)
	quLATi_mean, quLATi_sigma = findSpatioMeanAndSigma(bins, binEdges, figEstquLATi, nonEmpty)

	plt.subplot(421, title='Ground truth image'), plt.imshow(figTruth)
	plt.subplot(422, title='Ground truth color histograms'),
	plt.plot(np.sum(np.sum(true_hist, axis=1), axis=1), 'r')
	plt.plot(np.sum(np.sum(true_hist, axis=0), axis=1), 'g')
	plt.plot(np.sum(np.sum(true_hist, axis=0), axis=0), 'b')
	plt.xlim([0,bins])
	plt.subplot(423, title='MAGIC-LAT image'), plt.imshow(figEst)
	plt.subplot(424, title='MAGIC-LAT color histograms'),
	plt.plot(np.sum(np.sum(magic_hist, axis=1), axis=1), 'r')
	plt.plot(np.sum(np.sum(magic_hist, axis=0), axis=1), 'g')
	plt.plot(np.sum(np.sum(magic_hist, axis=0), axis=0), 'b')
	plt.xlim([0,bins])
	plt.subplot(425, title='GPR image'), plt.imshow(figEstGPR)
	plt.subplot(426, title='GPR color histograms'),
	plt.plot(np.sum(np.sum(gpr_hist, axis=1), axis=1), 'r')
	plt.plot(np.sum(np.sum(gpr_hist, axis=0), axis=1), 'g')
	plt.plot(np.sum(np.sum(gpr_hist, axis=0), axis=0), 'b')
	plt.xlim([0,bins])
	plt.subplot(427, title='quLATi image'), plt.imshow(figEstquLATi)
	plt.subplot(428, title='quLATi color histograms'),
	plt.plot(np.sum(np.sum(quLATi_hist, axis=1), axis=1), 'r')
	plt.plot(np.sum(np.sum(quLATi_hist, axis=0), axis=1), 'g')
	plt.plot(np.sum(np.sum(quLATi_hist, axis=0), axis=0), 'b')
	plt.xlim([0,bins])

	plt.tight_layout()
	plt.savefig(os.path.join(outDir, 'colorHist'+outputFileSuffix))
	# plt.show()

	plt.subplot(421, title='Ground truth image'), plt.imshow(figTruth)
	plt.subplot(422, title='Ground truth flattened histogram'),
	plt.plot(true_hist_flat)
	plt.xlim([0,true_hist_flat.shape[0]])
	plt.subplot(423, title='MAGIC-LAT image'), plt.imshow(figEst)
	plt.subplot(424, title='MAGIC-LAT flattened histogram'),
	plt.plot(magic_hist_flat)
	plt.xlim([0,true_hist_flat.shape[0]])
	plt.subplot(425, title='GPR image'), plt.imshow(figEstGPR)
	plt.subplot(426, title='GPR flattened histogram'),
	plt.plot(gpr_hist_flat)
	plt.xlim([0,true_hist_flat.shape[0]])
	plt.subplot(427, title='quLATi image'), plt.imshow(figEstquLATi)
	plt.subplot(428, title='quLATi flattened histogram'),
	plt.plot(quLATi_hist_flat)
	plt.xlim([0,true_hist_flat.shape[0]])

	plt.tight_layout()
	plt.savefig(os.path.join(outDir, 'hist'+outputFileSuffix))
	# plt.show()

	magic_corr = cv2.compareHist(true_hist_flat, magic_hist_flat, cv2.HISTCMP_CORREL)
	gpr_corr = cv2.compareHist(true_hist_flat, gpr_hist_flat, cv2.HISTCMP_CORREL)
	quLATi_corr = cv2.compareHist(true_hist_flat, quLATi_hist_flat, cv2.HISTCMP_CORREL)

	magic_spatiocorr = spatioCorr(bins, nonEmpty, true_mean, true_sigma, true_hist, 
		magic_mean, magic_sigma, magic_hist)
	
	gpr_spatiocorr = spatioCorr(bins, nonEmpty, true_mean, true_sigma, true_hist, 
		gpr_mean, gpr_sigma, gpr_hist)

	quLATi_spatiocorr = spatioCorr(bins, nonEmpty, true_mean, true_sigma, true_hist, 
		quLATi_mean, quLATi_sigma, quLATi_hist)

	return magic_spatiocorr, magic_corr, gpr_spatiocorr, gpr_corr, quLATi_spatiocorr, quLATi_corr



def createTestPointImages(vertices, faces, MINLAT, MAXLAT, outDir,
	coords, trueVals, magicVals, gprVals=[], quLATiVals=[]):

	mesh = Mesh([vertices, faces], c='black')

	"""
	Ground truth (test points only)
	"""
	plotSaveTestPoints(mesh, coords, trueVals,
		MINLAT, MAXLAT, outDir, fileprefix='true')

	"""
	MAGIC-LAT estimate (test points only)
	"""
	plotSaveTestPoints(mesh, coords, magicVals,
		MINLAT, MAXLAT, outDir, fileprefix='magic')

	"""
	GPR estimate (test points only)
	"""
	if len(gprVals) > 0:
		plotSaveTestPoints(mesh, coords, gprVals,
			MINLAT, MAXLAT, outDir, fileprefix='gpr')

	"""
	quLATi estimate (test points only)
	"""
	if len(quLATiVals) > 0:
		plotSaveTestPoints(mesh, coords, quLATiVals,
			MINLAT, MAXLAT, outDir, fileprefix='quLATi')


def plotSaveTestPoints(mesh, TstCoord, TstVal,
	MINLAT, MAXLAT, outDir, fileprefix):

	# vplt = Plotter(N=1, axes=0, offscreen=True)
	testPoints = Points(TstCoord, r=20).cmap('rainbow_r', TstVal, vmin=MINLAT, vmax=MAXLAT)

	elev = 0
	roll = 0
	azim = [0, 90, 180, 270]
	for a in azim:
		vplt = Plotter(N=1, axes=0, offscreen=True)
		vplt.show(mesh, testPoints, azimuth=a, elevation=elev, roll=roll, bg='black')
		vplt.screenshot(filename=os.path.join(outDir, fileprefix+'_elev{:g}azim{:g}.png'.format(elev, a)), returnNumpy=False)
		vplt.close()

	elev = [-90, 90]
	roll = 0
	azim = 0
	for e in elev:
		vplt = Plotter(N=1, axes=0, offscreen=True)
		vplt.show(mesh, testPoints, azimuth=azim, elevation=e, roll=roll, bg='black')
		vplt.screenshot(filename=os.path.join(outDir, fileprefix+'_elev{:g}azim{:g}.png'.format(e, azim)), returnNumpy=False)
		vplt.close()


def plotSaveEntire(mesh, latCoords, latVals, TrCoord, TrVal, latEst, 
	azimuth, elev, roll, MINLAT, MAXLAT, outDir, title, filename):

	vertices = mesh.points()

	verPoints = Points(latCoords, r=5).cmap('rainbow_r', latVals, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
	largeVerPoints = Points(latCoords, r=10).cmap('rainbow_r', latVals, vmin=MINLAT, vmax=MAXLAT).addScalarBar()

	vplt = Plotter(N=3, axes=9, offscreen=True)

	# Plot 0: Ground truth
	vplt.show(mesh, verPoints, 'all known points', azimuth=azimuth, elevation=elev, roll=roll, at=0)

	# Plot 1: Training points
	trainPoints = Points(TrCoord, r=5).cmap('rainbow_r', TrVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
	vplt.show(mesh, trainPoints, 'training points', at=1)

	# Plot 2: MAGIC-LAT output signal
	estPoints = Points(vertices, r=5).cmap('rainbow_r', latEst, vmin=MINLAT, vmax=MAXLAT).addScalarBar()

	# mesh.interpolateDataFrom(magicPoints, N=1).cmap('rainbow_r').addScalarBar()

	vplt.show(mesh, estPoints, largeVerPoints, 'interpolation result', title=title, at=2, interactive=True)
	vplt.screenshot(filename=os.path.join(outDir, filename), returnNumpy=False)
	vplt.close()


def getPerspective(patient):
	if patient == '033':
		elev = 0
		azimuth = 90
		roll = 0
	elif patient == '034':
		elev = 0
		azimuth = 120
		roll = -45
	elif patient == '035':
		elev = 0
		azimuth = 0
		roll = 0
	elif (patient == '037'):
		elev = 0
		azimuth = 160
		roll = 0
	else:
		print('no specified plot view for this patient')
		elev = 0
		azimuth = 0
		roll = 0
	return elev, azimuth, roll


def getModifiedSampList(latVals):
	M = len(latVals)

	sort_index = np.argsort(np.array(latVals))
	sortedLATVals = [latVals[i] for i in sort_index]
	pos = [int(abs(sortedLATVals[i] - max(sortedLATVals))) for i in range(M)]
	ratio = [(pos[i] / math.gcd(*pos)) for i in range(M)]

	sampLst = []
	for i in range(M):
		reps = ratio[i]
		idx = sort_index[i]
		if reps == 0.0:
			sampLst.append(idx)
		for r in range(int(reps)):
			sampLst.append(idx)
	# print(sampLst.count(latVals.index(min(latVals))))

	return sampLst


def isAnomalous(allLatCoord, allLatVal, k=6, d=5, thresh=50):
	# KD Tree to find the nearest mesh vertex
	coordKDtree = cKDTree(allLatCoord)
	[dist, nearestVers] = coordKDtree.query(allLatCoord, k=k)

	M = len(allLatCoord)
	anomalous = np.zeros(M)

	for i in range(M):
		verCoord = allLatCoord[i]
		verVal = allLatVal[i]

		neighbors = [nearestVers[i, n] for n in range(1,k) if dist[i,n] < d]

		adj = len(neighbors)

		cnt = 0
		for neighVer in neighbors:
			neighVal = allLatVal[neighVer]

			if abs(verVal - neighVal) > thresh:
				cnt += 1
			else:
				break

		# if (cnt >= (len(neighbors)-1) and len(neighbors) > 1):	# differs from all but 1 neighbor by >50ms and has at least 2 neighbors w/in 5mm
		if cnt > 1 and adj > 1:
			anomalous[i] = 1
			# print(cnt, adj)

			# print(verVal, [allLatVal[neighVer] for neighVer in neighbors])
	return anomalous


def delta_e(a, b):
	"""CIE 1976 """
	# compute difference
	diff = cv2.add(a,-b)

	# separate into L,A,B channel diffs
	diff_L = diff[:,:,0]
	diff_A = diff[:,:,1]
	diff_B = diff[:,:,2]

	# compute delta_e as mean over every pixel using equation from  
	# https://en.wikipedia.org/wiki/Color_difference#CIELAB_ΔE*
	delta_E = np.mean( np.sqrt(diff_L*diff_L + diff_A*diff_A + diff_B*diff_B) )

	return delta_E


def deltaE(trueVals, estVals, MINLAT, MAXLAT, cmap=cm.rainbow_r):
	norm = Normalize(vmin=MINLAT, vmax=MAXLAT)

	trueColors = cmap(norm(np.array([trueVals])))[:, :, :3]
	trueColors = cv2.cvtColor(trueColors.astype("float32") / 255, cv2.COLOR_RGB2LAB)

	estColors = cmap(norm(np.array(estVals)))[:, :, :3]
	estColors = estColors.reshape((1, max(estColors.shape), 3))
	estColors = cv2.cvtColor(estColors.astype("float32") / 255, cv2.COLOR_RGB2LAB)

	dE1976 = delta_e(trueColors, estColors)

	""" https://colour.readthedocs.io/en/latest/generated/colour.delta_E.html """
	dE2000 = np.mean(colour.delta_E(trueColors, estColors, method='CIE 2000'))

	return dE1976, dE2000

def mapSamps(IDX, COORD, coords, vals):
	"""
	Maps LAT sample values (vals) at coordinates (coords) not
	on the mesh to the nearest mesh coordinate in COORD.

	Returns:
	- latIdx, a list of mesh vertex indices with an LAT sample
	- latCoords, a list of corr. LAT vertex coordinates
	- latVals, a list of corr. LAT values

	"""
	n = len(IDX)	# number of vertices in the graph
	m = len(coords)			# number of signal samples

	# KD Tree to find the nearest mesh vertex
	coordKDtree = cKDTree(COORD)
	[_, nearestVer] = coordKDtree.query(coords, k=1)

	# find the vertices with an assigned (known) sample
	known = [False for i in range(n)]
	lat = [0 for i in range(n)]
	for sampPt in range(m):
		verIdx = nearestVer[sampPt]	# nearest vertex
		known[verIdx] = True
		lat[verIdx] = vals[sampPt]	# assign the value

	latIdx = [IDX[i] for i in range(n) if known[i] is True]
	latCoords = [COORD[i] for i in range(n) if known[i] is True]
	latVals = [lat[i] for i in range(n) if known[i] is True]

	return latIdx, latCoords, latVals





def calcMSE(sig, sigEst, multichannel=False):
	if multichannel:
		n = sig.shape[1]	# summing across the number of bins
		diffsq = (np.array(sigEst) - np.array(sig)) ** 2
		d = np.sum(diffsq[:, :, 1:], axis=2)	# (x' - x)^2 + (y' - y)^2 = d^2 (pixel distance)
		mean_err = np.sum(d, axis=1, keepdims=True)
		hist_err = np.sum(diffsq[:,:,0], axis=1, keepdims=True)
		return (1/n * hist_err, 1/n * mean_err)
	else:
		n = len(sig)

		err = [abs(sigEst[i] - sig[i]) for i in range(n)]
		err = np.array(err)

		mse = 1/n*np.sum(err ** 2)

	return mse


def calcMAE(sig, sigEst):
	delta = [abs(sig[i] - sigEst[i]) for i in range(len(sig))]
	delta = np.array(delta)
	return np.average(delta)



def calcPercError(sig, sigEst):
	n = len(sig)

	err = [abs(sigEst[i] - sig[i]) for i in range(n)]
	err = [1 for i in err if i > 10]
	err = np.array(err)

	return float(np.sum(err))/n


def calcNMSE(sig, sigEst, multichannel=False):
	if multichannel:
		print(sig.shape, sigEst.shape)
		err = (np.array(sigEst) - np.array(sig)) ** 2
		err = np.sum(err, axis=0, keepdims = True)
		meanvec = np.array(np.mean(sig, axis=0), ndmin=2)
		# sigPower = np.sum((np.array(sig) - meanvec), axis=0, keepdims = True)
		sigPower = np.sum(np.array(sig), axis=0, keepdims=True)

		nmse = err / sigPower
	else:
		n = len(sig)

		err = [abs(sigEst[i] - sig[i]) for i in range(n)]
		err = np.array(err)

		sigKnown = [sig[i] for i in range(n)]
		sigPower = np.sum((np.array(sigKnown) - np.mean(sigKnown)) ** 2)

		nmse = np.sum(err ** 2)/sigPower

	return nmse



def calcNRMSE(sig, sigEst):
	n = len(sig)

	err = [abs(sigEst[i] - sig[i]) for i in range(n)]
	sqErr = np.array(err) ** 2

	rmse = (np.sum(sqErr)/n)**(1/2)
	nRMSE = 100*rmse/(np.max(sig) - np.min(sig))

	return nRMSE



def calcSNR(sig, sigEst):
	n = len(sig)
	err = [abs(sigEst[i] - sig[i]) for i in range(n)]
	err = np.array(err)

	sigKnown = [sig[i] for i in range(n)]
	sigPower = np.sum((np.array(sigKnown) - np.mean(sigKnown)) ** 2)

	snr = 20*np.log10(sigPower/np.sum(err ** 2))

	return snr



def compute_metrics(sig, sigEst):

	nmse = calcNMSE(sig, sigEst)
	snr = calcSNR(sig, sigEst)
	mae = calcMAE(sig, sigEst)
	nrmse = calcNRMSE(sig, sigEst)

	return nmse, snr, mae, nrmse



def plotTrainTestVertices(coordinateMatrix, connectivityMatrix, lat, latTrI, latTstI, nm):
	TrainVerCoord = {}
	TrainVerLAT = {}
	for i in latTrI:
		TrainVerCoord[i] = lat[i]['coord']
		TrainVerLAT[i] = lat[i]['val']

	TestVerCoord = {}
	TestVerLAT = {}
	for i in latTstI:
		TestVerCoord[i] = lat[i]['coord']
		TestVerLAT[i] = lat[i]['val']

	TrainCoordList = np.array(list(TrainVerCoord.values()))
	TestCoordList = np.array(list(TestVerCoord.values()))

	fig, ax = plt.subplots(figsize=(16, 8), subplot_kw=dict(projection="3d"))

	triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)
	ax.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	ax.scatter(TrainCoordList[:,0], TrainCoordList[:,1], TrainCoordList[:,2], c='blue', s = 20)
	ax.scatter(TestCoordList[:,0], TestCoordList[:,1], TestCoordList[:,2], c='red', s = 20)
	ax.set_title(nm)

	plt.show()


def plotTrainTestResult(coordinateMatrix, connectivityMatrix, lat, latVer, latVals, latTrI, latTstI, yhat):
	TrainVerCoord = {}
	TrainVerLAT = {}
	for i in latTrI:
		TrainVerCoord[i] = lat[i]['coord']
		TrainVerLAT[i] = lat[i]['val']

	TestVerCoord = {}
	TestVerLAT = {}
	for i in latTstI:
		TestVerCoord[i] = lat[i]['coord']
		TestVerLAT[i] = yhat[i]

	TrainCoordList = np.array(list(TrainVerCoord.values()))
	TestCoordList = np.array(list(TestVerCoord.values()))

	TrainValList = np.array(list(TrainVerLAT.values()))
	TestValList = np.array(list(TestVerLAT.values()))

	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 8), subplot_kw=dict(projection="3d"))
	axes = ax.flatten()

	triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

	thisAx = axes[0]
	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(latVer[:,0], latVer[:,1], latVer[:,2], c=latVals, cmap='rainbow_r', s = 10)
	thisAx.set_title('Ground truth')

	thisAx = axes[1]
	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(TrainCoordList[:,0], TrainCoordList[:,1], TrainCoordList[:,2], c=TrainValList, cmap='rainbow_r', s = 10)
	thisAx.set_title('Given (y)')

	thisAx = axes[2]
	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(TestCoordList[:,0], TestCoordList[:,1], TestCoordList[:,2], c=TestValList, cmap='rainbow_r', s = 10)
	thisAx.set_title('Test Output')
	cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height])
	plt.colorbar(pos, cax=cax) # Similar to fig.colorbar(im, cax = cax)

	plt.show()


def pltEigenvalues(lambdas):
	N = len(lambdas)
	plt.scatter(range(0, N), lambdas)
	plt.xlabel('Eigenvalue Index')
	plt.ylabel('Eigenvalue')
	plt.title('Graph Laplacian Eigenvalues')
	plt.show()


def getAdjMatrixCotan(coordinateMatrix, edges, triangles):
	N = len(coordinateMatrix)
	A = np.zeros((N, N))

	for i in range(len(edges)):

		e = edges[i]
		adj_tri = triangles[i]

		w_ij = 0

		for tri in adj_tri:
			idx0 = int(tri[0])
			idx1 = int(tri[1])
			idx2 = int(tri[2])

			pt0 = coordinateMatrix[idx0, :]
			pt1 = coordinateMatrix[idx1, :]
			pt2 = coordinateMatrix[idx2, :]

			l_a = norm(pt1 - pt0)
			l_b = norm(pt2 - pt0)
			l_c = norm(pt2 - pt1)

			s = (l_a + l_b + l_c)/2

			a = math.sqrt(s*(s-l_a)*(s-l_b)*(s-l_c))

			if set([idx0, idx1]) == e:
				w_ij = w_ij + (-l_a**2 + l_b**2 + l_c**2)/(8*a)
			elif set([idx1, idx2]) == e:
				w_ij = w_ij + (l_a**2 + l_b**2 - l_c**2)/(8*a)
			elif set([idx0, idx2]) == e:
				w_ij = w_ij + (l_a**2 - l_b**2 + l_c**2)/(8*a)
			else:
				print('unable to identify edge')
				exit()

		e = list(e)
		v_i = e[0]
		v_j = e[1]

		A[v_i, v_j] = w_ij
		A[v_j, v_i] = w_ij

	return A


def getAdjMatrixExp(coordinateMatrix, edges, triangles):
	N = len(coordinateMatrix)
	A = np.zeros((N, N))

	d = 0
	cnt = 0
	for i in range(len(coordinateMatrix)):
		[x1, y1, z1] = coordinateMatrix[i, :]
		for j in range(len(coordinateMatrix)):
			[x2, y2, z2] = coordinateMatrix[j, :]
			d += math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
			cnt += 1
	d = d/cnt

	# d = 0
	# for i in range(len(edges)):
	# 	e = edges[i]

	# 	e = list(e)
	# 	v_i = e[0]
	# 	v_j = e[1]

	# 	[x1, y1, z1] = coordinateMatrix[v_i, :]
	# 	[x2, y2, z2] = coordinateMatrix[v_j, :]

	# 	d += math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
	# d = d/len(edges)

	for i in range(len(edges)):

		e = edges[i]

		e = list(e)
		v_i = e[0]
		v_j = e[1]

		[x1, y1, z1] = coordinateMatrix[v_i, :]
		[x2, y2, z2] = coordinateMatrix[v_j, :]

		# w_ij = 1/math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
		d2_ij = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
		w_ij = math.exp(-d2_ij/d**2)

		A[v_i, v_j] = w_ij
		A[v_j, v_i] = w_ij

	return A


def pltAdjMatrix(A, first, numV, title):

	end = first + numV + 1;

	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8))
	pos = ax.imshow(A[first:end, first:end], cmap='Blues', interpolation=None)
	plt.xticks(np.arange(first, end, step=1))
	plt.yticks(np.arange(first, end, step=1))
	plt.title(title + '\nVertices ' + str(first) + ' - ' + str(end-1))
	# plt.colorbar(hm)
	# cax = fig.add_axes([ax.get_position().x0,ax.get_position().y0-0.1,ax.get_position().width,0.01])
	cax = fig.add_axes([ax.get_position().x1+0.03,ax.get_position().y0,0.01,ax.get_position().height])
	plt.colorbar(pos, cax=cax, label='weight')
	# plt.show()
