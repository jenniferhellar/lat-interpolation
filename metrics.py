
"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""

import os

import numpy as np
import math
import random

# plotting packages
from vedo import *

from matplotlib import pyplot as plt

# loading images
import cv2

# Gaussian process regression interpolation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT

from js import *


from utils import *
from const import *
from magicLAT import *

# from scipy.spatial.distance import jensenshannon

from scipy.stats import multivariate_normal



"""
p033 = 9
p034 = 14
p035 = 18
p037 = 21
"""
PATIENT_MAP				=		9

NUM_TRAIN_SAMPS 		= 		100
EDGE_THRESHOLD			=		50

outDir				 	=		'results_wip'

""" Read the files """
meshFile = meshNames[PATIENT_MAP]
latFile = latNames[PATIENT_MAP]
nm = meshFile[0:-5]
patient = nm[7:10]

print('Reading files for ' + nm + ' ...\n')
[vertices, faces] = readMesh(os.path.join(dataDir, meshFile))
[OrigLatCoords, OrigLatVals] = readLAT(os.path.join(dataDir, latFile))

n = len(vertices)

mapIdx = [i for i in range(n)]
mapCoord = [vertices[i] for i in mapIdx]

allLatIdx, allLatCoord, allLatVal = mapSamps(mapIdx, mapCoord, OrigLatCoords, OrigLatVals)

M = len(allLatIdx)

mesh = Mesh([vertices, faces])
# mesh.backColor('white').lineColor('black').lineWidth(0.25)
mesh.c('grey')

origLatPoints = Points(OrigLatCoords, r=10).cmap('rainbow_r', OrigLatVals, vmin=np.min(OrigLatVals), vmax=np.max(OrigLatVals)).addScalarBar()
latPoints = Points(allLatCoord, r=10).cmap('rainbow_r', allLatVal, vmin=np.min(allLatVal), vmax=np.max(allLatVal)).addScalarBar()

# KD Tree to find the nearest mesh vertex
k = 6
coordKDtree = cKDTree(allLatCoord)
[dist, nearestVers] = coordKDtree.query(allLatCoord, k=k)

anomalous = np.zeros(M)

for i in range(M):
	verCoord = allLatCoord[i]
	verVal = allLatVal[i]

	neighbors = [nearestVers[i, n] for n in range(1,k) if dist[i,n] < 5]

	adj = len(neighbors)

	cnt = 0
	for neighVer in neighbors:
		neighVal = allLatVal[neighVer]

		if abs(verVal - neighVal) > 50:
			cnt += 1
		else:
			break

	# if (cnt >= (len(neighbors)-1) and len(neighbors) > 1):	# differs from all but 1 neighbor by >50ms and has at least 2 neighbors w/in 5mm
	if cnt > 1 and adj > 1:
		anomalous[i] = 1
		# print(cnt, adj)

		# print(verVal, [allLatVal[neighVer] for neighVer in neighbors])

numPtsIgnored = np.sum(anomalous)

latIdx = [allLatIdx[i] for i in range(M) if anomalous[i] == 0]
latCoords = [allLatCoord[i] for i in range(M) if anomalous[i] == 0]
latVals = [allLatVal[i] for i in range(M) if anomalous[i] == 0]

M = len(latIdx)

print('{:<20}{:g}'.format('n', n))
print('{:<20}{:g}/{:g}'.format('m', NUM_TRAIN_SAMPS, M))
print('{:<20}{:g}\n'.format('ignored', numPtsIgnored))
# exit()

mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]

edgeFile = 'E_p{}.npy'.format(patient)
if not os.path.isfile(edgeFile):
	[edges, TRI] = edgeMatrix(vertices, faces)

	print('Writing edge matrix to file...')
	with open(edgeFile, 'wb') as fid:
		np.save(fid, edges)
else:
	edges = np.load(edgeFile, allow_pickle=True)

if not os.path.isdir(outDir):
	os.makedirs(outDir)

sampLst = [i for i in range(M)]


tr_i = random.sample(sampLst, NUM_TRAIN_SAMPS)
tst_i = [i for i in sampLst if i not in tr_i]

# get vertex indices of labelled/unlabelled nodes
TrIdx = sorted(np.take(latIdx, tr_i))
TstIdx = sorted(np.take(latIdx, tst_i))

# get vertex coordinates
TrCoord = [vertices[i] for i in TrIdx]
TstCoord = [vertices[i] for i in TstIdx]

# get mapLAT signal values
TrVal = [mapLAT[i] for i in TrIdx]
TstVal = [mapLAT[i] for i in TstIdx]


""" MAGIC-LAT estimate """
latEst = magicLAT(vertices, faces, edges, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)

""" Create GPR kernel and regressor """
gp_kernel = RBF(length_scale=0.01) + RBF(length_scale=0.1) + RBF(length_scale=1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)

""" GPR estimate """
# fit the GPR with training samples
gpr.fit(TrCoord, TrVal)

# predict the entire signal
latEstGPR = gpr.predict(vertices, return_std=False)


# For colorbar ranges
MINLAT = math.floor(min(allLatVal)/10)*10
MAXLAT = math.ceil(max(allLatVal)/10)*10

elev = 0
azimuth = 120
roll = -45

verPoints = Points(latCoords, r=10).cmap('rainbow_r', latVals, vmin=MINLAT, vmax=MAXLAT).addScalarBar()

"""
Figure 0: Ground truth (entire), training points, and MAGIC-LAT (entire)
"""
vplt = Plotter(N=3, axes=9, offscreen=True)

# Plot 0: Ground truth
vplt.show(mesh, verPoints, 'all known points', azimuth=azimuth, elevation=elev, roll=roll, at=0)

# Plot 1: Training points
trainPoints = Points(TrCoord, r=10).cmap('rainbow_r', TrVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
vplt.show(mesh, trainPoints, 'training points', at=1)

# Plot 2: MAGIC-LAT output signal
magicPoints = Points(vertices, r=10).cmap('rainbow_r', latEst, vmin=MINLAT, vmax=MAXLAT).addScalarBar()

vplt.show(mesh, magicPoints, 'interpolation result', title='MAGIC-LAT', at=2, interactive=True)
vplt.screenshot(filename=os.path.join(outDir, 'magic.png'), returnNumpy=False)
vplt.close()


"""
Figure 1: Ground truth (entire), training points, and GPR (entire)
"""
vplt = Plotter(N=3, axes=9, offscreen=True)

# Plot 0: Ground truth
vplt.show(mesh, verPoints, 'all known points', azimuth=azimuth, elevation=elev, roll=roll, at=0)
# Plot 1: Training points
vplt.show(mesh, trainPoints, 'training points', at=1)
# Plot 2: GPR output signal
gprPoints = Points(vertices, r=10).cmap('rainbow_r', latEstGPR, vmin=MINLAT, vmax=MAXLAT).addScalarBar()

vplt.show(mesh, gprPoints, 'interpolation result', title='GPR', at=2, interactive=True)
vplt.screenshot(filename=os.path.join(outDir, 'gpr.png'), returnNumpy=False)
vplt.close()

# mesh.interpolateDataFrom(pts, N=1).cmap('rainbow_r').addScalarBar()
elev = 0
azim = [-60, 30, 120, 210]
roll = -45
whitemesh = Mesh([vertices, faces], c='black')
"""
Figure 2: Ground truth (test points only - for ssim)
"""
vplt = Plotter(N=1, axes=0, offscreen=True)
testPoints = Points(TstCoord, r=20).cmap('rainbow_r', TstVal, vmin=MINLAT, vmax=MAXLAT)
for a in azim:
	vplt.show(whitemesh, testPoints, azimuth=a, elevation=elev, roll=roll, title='true, azimuth={:g}'.format(a), bg='black')
	vplt.screenshot(filename=os.path.join(outDir, 'true{:g}.png'.format(a)), returnNumpy=False)

vplt.close()

"""
Figure 3: MAGIC-LAT estimate (test points only - for ssim)
"""
vplt = Plotter(N=1, axes=0, offscreen=True)
testEst = Points(TstCoord, r=20).cmap('rainbow_r', latEst[TstIdx], vmin=MINLAT, vmax=MAXLAT)

for a in azim:
	vplt.show(whitemesh, testEst, azimuth=a, elevation=elev, roll=roll, title='MAGIC-LAT, azimuth={:g}'.format(a), bg='black')
	vplt.screenshot(filename=os.path.join(outDir, 'estimate{:g}.png'.format(a)), returnNumpy=False)

vplt.close()

"""
Figure 4: GPR estimate (test points only - for ssim)
"""
vplt = Plotter(N=1, axes=0, offscreen=True)
testEstGPR = Points(TstCoord, r=20).cmap('rainbow_r', latEstGPR[TstIdx], vmin=MINLAT, vmax=MAXLAT)

for a in azim:
	vplt.show(whitemesh, testEstGPR, azimuth=a, elevation=elev, roll=roll, title='GPR, azimuth={:g}'.format(a), bg='black')
	vplt.screenshot(filename=os.path.join(outDir, 'estimateGPR{:g}.png'.format(a)), returnNumpy=False)

vplt.close()


"""
Figure 5: quLATi estimate (test points only for ssim)
"""
# TODO


"""
Error metrics
"""

nmse = calcNMSE(TstVal, latEst[TstIdx])
nmseGPR = calcNMSE(TstVal, latEstGPR[TstIdx])

mae = calcMAE(TstVal, latEst[TstIdx])
maeGPR = calcMAE(TstVal, latEstGPR[TstIdx])

magic_spatio_corr = []
gpr_spatio_corr = []

magic_spatio_chi = []
gpr_spatio_chi = []

magic_corr = []
gpr_corr = []

magic_chi = []
gpr_chi = []

bins = 16
binEdges = [i*256/bins for i in range(bins+1)]

for a in azim:
	img = cv2.imread(os.path.join(outDir, 'true{:g}.png'.format(a)), cv2.IMREAD_GRAYSCALE)
	n_black_px = np.sum(img == 0)
	# numpx = np.sum(img > 0)

	figTruth = cv2.imread(os.path.join(outDir, 'true{:g}.png'.format(a)))
	figTruth = cv2.cvtColor(figTruth, cv2.COLOR_BGR2RGB)
	
	true_hist = cv2.calcHist([figTruth], [0, 1, 2], None, [bins, bins, bins],
		[0, 256, 0, 256, 0, 256])
	true_hist[0, 0, 0] -= n_black_px
	true_hist_flat = cv2.normalize(true_hist, true_hist).flatten()

	figEst = cv2.imread(os.path.join(outDir, 'estimate{:g}.png'.format(a)))
	figEst = cv2.cvtColor(figEst, cv2.COLOR_BGR2RGB)
	
	magic_hist = cv2.calcHist([figEst], [0, 1, 2], None, [bins, bins, bins],
		[0, 256, 0, 256, 0, 256])
	magic_hist[0, 0, 0] -= n_black_px
	magic_hist_flat = cv2.normalize(magic_hist, magic_hist).flatten()

	figEstGPR = cv2.imread(os.path.join(outDir, 'estimateGPR{:g}.png'.format(a)))
	figEstGPR = cv2.cvtColor(figEstGPR, cv2.COLOR_BGR2RGB)
	
	gpr_hist = cv2.calcHist([figEstGPR], [0, 1, 2], None, [bins, bins, bins],
		[0, 256, 0, 256, 0, 256])
	gpr_hist[0, 0, 0] -= n_black_px
	gpr_hist_flat = cv2.normalize(gpr_hist, gpr_hist).flatten()

	true_mean = np.zeros((bins, bins, bins, 2))
	true_sigma = np.zeros((bins, bins, bins, 2, 2))

	magic_mean = np.zeros((bins, bins, bins, 2))
	magic_sigma = np.zeros((bins, bins, bins, 2, 2))

	gpr_mean = np.zeros((bins, bins, bins, 2))
	gpr_sigma = np.zeros((bins, bins, bins, 2, 2))

	for r_i, r_v in enumerate(binEdges):
		for g_i, g_v in enumerate(binEdges):
			for b_i, b_v in enumerate(binEdges):
				if r_v < 256 and g_v < 256 and b_v < 256:
					if true_hist[r_i, g_i, b_i] > 0 or magic_hist[r_i, g_i, b_i] > 0 or gpr_hist[r_i, g_i, b_i] > 0:
						true_pxls = np.column_stack(np.where((figTruth[:, :, 0] >= r_v) & (figTruth[:, :, 0] < binEdges[r_i + 1]) & \
							(figTruth[:, :, 1] >= g_v) & (figTruth[:, :, 1] < binEdges[g_i + 1]) & \
							(figTruth[:, :, 2] >= b_v) & (figTruth[:, :, 2] < binEdges[b_i + 1])))
						if true_pxls.shape[0] == 0:
							true_sigma[r_i, g_i, b_i] = np.array([[1, 0], [0, 1]])
						else:
							true_mean[r_i, g_i, b_i] = np.mean(true_pxls, axis=0)
							true_sigma[r_i, g_i, b_i] = np.cov(true_pxls, rowvar=0)
				
						pxls = np.column_stack(np.where((figEst[:, :, 0] >= r_v) & (figEst[:, :, 0] < binEdges[r_i + 1]) & \
									(figEst[:, :, 1] >= g_v) & (figEst[:, :, 1] < binEdges[g_i + 1]) & \
									(figEst[:, :, 2] >= b_v) & (figEst[:, :, 2] < binEdges[b_i + 1])))
						if pxls.shape[0] == 0:
							magic_sigma[r_i, g_i, b_i] = np.array([[1, 0], [0, 1]])
						else:
							magic_mean[r_i, g_i, b_i] = np.mean(pxls, axis=0)
							magic_sigma[r_i, g_i, b_i] = np.cov(pxls, rowvar=0)

						pxls = np.column_stack(np.where((figEstGPR[:, :, 0] >= r_v) & (figEstGPR[:, :, 0] < binEdges[r_i + 1]) & \
									(figEstGPR[:, :, 1] >= g_v) & (figEstGPR[:, :, 1] < binEdges[g_i + 1]) & \
									(figEstGPR[:, :, 2] >= b_v) & (figEstGPR[:, :, 2] < binEdges[b_i + 1])))
						if pxls.shape[0] == 0:
							gpr_sigma[r_i, g_i, b_i] = np.array([[1, 0], [0, 1]])
						else:
							gpr_mean[r_i, g_i, b_i] = np.mean(pxls, axis=0)
							gpr_sigma[r_i, g_i, b_i] = np.cov(pxls, rowvar=0)

	plt.subplot(321, title='Ground truth image'), plt.imshow(figTruth)
	plt.subplot(322, title='Ground truth color histograms'),
	plt.plot(np.sum(np.sum(true_hist, axis=1), axis=1), 'r')
	plt.plot(np.sum(np.sum(true_hist, axis=0), axis=1), 'g')
	plt.plot(np.sum(np.sum(true_hist, axis=0), axis=0), 'b')
	plt.xlim([0,bins])
	plt.subplot(323, title='MAGIC-LAT image'), plt.imshow(figEst)
	plt.subplot(324, title='MAGIC-LAT color histograms'),
	plt.plot(np.sum(np.sum(magic_hist, axis=1), axis=1), 'r')
	plt.plot(np.sum(np.sum(magic_hist, axis=0), axis=1), 'g')
	plt.plot(np.sum(np.sum(magic_hist, axis=0), axis=0), 'b')
	plt.xlim([0,bins])
	plt.subplot(325, title='GPR image'), plt.imshow(figEstGPR)
	plt.subplot(326, title='GPR color histograms'),
	plt.plot(np.sum(np.sum(gpr_hist, axis=1), axis=1), 'r')
	plt.plot(np.sum(np.sum(gpr_hist, axis=0), axis=1), 'g')
	plt.plot(np.sum(np.sum(gpr_hist, axis=0), axis=0), 'b')
	plt.xlim([0,bins])

	plt.tight_layout()
	plt.savefig(os.path.join(outDir, 'colorHist{:g}.png'.format(a)))
	# plt.show()

	plt.subplot(321, title='Ground truth image'), plt.imshow(figTruth)
	plt.subplot(322, title='Ground truth flattened histogram'),
	plt.plot(true_hist_flat)
	plt.xlim([0,true_hist_flat.shape[0]])
	plt.subplot(323, title='MAGIC-LAT image'), plt.imshow(figEst)
	plt.subplot(324, title='MAGIC-LAT flattened histogram'),
	plt.plot(magic_hist_flat)
	plt.xlim([0,true_hist_flat.shape[0]])
	plt.subplot(325, title='GPR image'), plt.imshow(figEstGPR)
	plt.subplot(326, title='GPR flattened histogram'),
	plt.plot(gpr_hist_flat)
	plt.xlim([0,true_hist_flat.shape[0]])

	plt.tight_layout()
	plt.savefig(os.path.join(outDir, 'hist{:g}.png'.format(a)))
	# plt.show()

	magic_corr.append(cv2.compareHist(true_hist_flat, magic_hist_flat, cv2.HISTCMP_CORREL))
	gpr_corr.append(cv2.compareHist(true_hist_flat, gpr_hist_flat, cv2.HISTCMP_CORREL))

	magic_chi.append(cv2.compareHist(true_hist_flat, magic_hist_flat, cv2.HISTCMP_CHISQR_ALT))
	gpr_chi.append(cv2.compareHist(true_hist_flat, gpr_hist_flat, cv2.HISTCMP_CHISQR_ALT))

	true_hist = cv2.normalize(true_hist, true_hist)
	magic_hist = cv2.normalize(magic_hist, magic_hist)
	gpr_hist = cv2.normalize(gpr_hist, gpr_hist)

	magic_similarity = 0
	gpr_similarity = 0

	for i in range(bins):
		for j in range(bins):
			for k in range(bins):
				magic_js = js(true_mean[i,j,k], magic_mean[i,j,k], true_sigma[i,j,k], magic_sigma[i,j,k], true_hist[i,j,k], magic_hist[i,j,k])
				
				num = (true_hist[i,j,k] - np.mean(true_hist)) * (magic_hist[i,j,k] - np.mean(magic_hist))
				denom = math.sqrt(np.sum((true_hist - np.mean(true_hist)) ** 2) * np.sum((magic_hist - np.mean(magic_hist)) ** 2))
				magic_similarity += (num / denom) * math.exp(-1 * magic_js)

				gpr_js = js(true_mean[i,j,k], gpr_mean[i,j,k], true_sigma[i,j,k], gpr_sigma[i,j,k], true_hist[i,j,k], gpr_hist[i,j,k])

				num = (true_hist[i,j,k] - np.mean(true_hist)) * (gpr_hist[i,j,k] - np.mean(gpr_hist))
				denom = math.sqrt(np.sum((true_hist - np.mean(true_hist)) ** 2) * np.sum((gpr_hist - np.mean(gpr_hist)) ** 2))
				gpr_similarity += (num / denom) * math.exp(-1 * gpr_js)

				# Intersection of histograms
				# magic_similarity += min(true_hist[i, j, k], magic_hist[i, j, k])*math.exp(-1 * magic_js[i, j, k])
				# gpr_similarity += min(true_hist[i, j, k], gpr_hist[i, j, k])*math.exp(-1 * gpr_js[i, j, k])

	# print('{:<20.6f}{:<20.6f}\n'.format(magic_similarity, gpr_similarity))
	magic_spatio_corr.append(magic_similarity)
	gpr_spatio_corr.append(gpr_similarity)


with open(os.path.join(outDir, 'metrics_ex.txt'), 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}\n\n'.format('Metric', 'MAGIC-LAT', 'GPR'))
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('NMSE', nmse, nmseGPR))
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('MAE', mae, maeGPR))

	fid.write('\nColor-Based\n')
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('Histogram Corr.', np.mean(magic_corr), np.mean(gpr_corr)))
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('Histogram Chi', np.mean(magic_chi), np.mean(gpr_chi)))

	fid.write('\n')
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('Spatiogram Corr.', np.mean(magic_spatio_corr), np.mean(gpr_spatio_corr)))
