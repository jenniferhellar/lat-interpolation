
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

from qulati import gpmi, eigensolver



"""
p033 = 9
p034 = 14
p035 = 18
p037 = 21
"""
PATIENT_MAP				=		9

NUM_TRAIN_SAMPS 		= 		100
EDGE_THRESHOLD			=		50

NUM_TEST_REPEATS 		= 		20

outDir				 	=		'results_repeated_wip'

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

"""
Solve the eigenproblem
""" 

Q, V, gradV, centroids = eigensolver(vertices, np.array(faces), holes = 0, layers = 10, num = 256)

# model with reduced rank efficiency
model = gpmi.Matern(vertices, np.array(faces), Q, V, gradV, JAX = False)

sampLst = [i for i in range(M)]

magicNMSE = [0 for i in range(NUM_TEST_REPEATS)]
magicMAE = [0 for i in range(NUM_TEST_REPEATS)]
magicHistCorr = [0 for i in range(NUM_TEST_REPEATS)]
magicSpatioCorr = [0 for i in range(NUM_TEST_REPEATS)]

gprNMSE = [0 for i in range(NUM_TEST_REPEATS)]
gprMAE = [0 for i in range(NUM_TEST_REPEATS)]
gprHistCorr = [0 for i in range(NUM_TEST_REPEATS)]
gprSpatioCorr = [0 for i in range(NUM_TEST_REPEATS)]

quLATiNMSE = [0 for i in range(NUM_TEST_REPEATS)]
quLATiMAE = [0 for i in range(NUM_TEST_REPEATS)]
quLATiHistCorr = [0 for i in range(NUM_TEST_REPEATS)]
quLATiSpatioCorr = [0 for i in range(NUM_TEST_REPEATS)]

for test in range(NUM_TEST_REPEATS):
	
	print('test #{:g} of {:g}.'.format(test + 1, NUM_TEST_REPEATS))

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

	""" quLATi estimate """
	obs = np.array(TrVal)
	quLATi_vertices = np.array(TrIdx)

	model.set_data(obs, quLATi_vertices)
	model.kernelSetup(smoothness = 3./2.)

	# optimize the nugget
	model.optimize(nugget = None, restarts = 5)

	pred_mean, pred_stdev = model.posterior(pointwise = True)

	latEstquLATi = pred_mean[0:vertices.shape[0]]


	# For colorbar ranges
	MINLAT = math.floor(min(allLatVal)/10)*10
	MAXLAT = math.ceil(max(allLatVal)/10)*10

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
	vplt = Plotter(N=1, axes=0, offscreen=True)
	testEstquLATi = Points(TstCoord, r=20).cmap('rainbow_r', latEstquLATi[TstIdx], vmin=MINLAT, vmax=MAXLAT)

	for a in azim:
		vplt.show(whitemesh, testEstquLATi, azimuth=a, elevation=elev, roll=roll, title='quLATi, azimuth={:g}'.format(a), bg='black')
		vplt.screenshot(filename=os.path.join(outDir, 'estimatequLATi{:g}.png'.format(a)), returnNumpy=False)

	vplt.close()


	"""
	Error metrics
	"""

	nmse = calcNMSE(TstVal, latEst[TstIdx])
	nmseGPR = calcNMSE(TstVal, latEstGPR[TstIdx])
	nmsequLATi = calcNMSE(TstVal, latEstquLATi[TstIdx])

	mae = calcMAE(TstVal, latEst[TstIdx])
	maeGPR = calcMAE(TstVal, latEstGPR[TstIdx])
	maequLATi = calcMAE(TstVal, latEstquLATi[TstIdx])

	magic_spatio_corr = []
	gpr_spatio_corr = []
	quLATi_spatio_corr = []

	magic_corr = []
	gpr_corr = []
	quLATi_corr = []

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

		figEstquLATi = cv2.imread(os.path.join(outDir, 'estimatequLATi{:g}.png'.format(a)))
		figEstquLATi = cv2.cvtColor(figEstquLATi, cv2.COLOR_BGR2RGB)
		
		quLATi_hist = cv2.calcHist([figEstquLATi], [0, 1, 2], None, [bins, bins, bins],
			[0, 256, 0, 256, 0, 256])
		quLATi_hist[0, 0, 0] -= n_black_px
		quLATi_hist_flat = cv2.normalize(quLATi_hist, quLATi_hist).flatten()

		true_mean = np.zeros((bins, bins, bins, 2))
		true_sigma = np.zeros((bins, bins, bins, 2, 2))

		magic_mean = np.zeros((bins, bins, bins, 2))
		magic_sigma = np.zeros((bins, bins, bins, 2, 2))

		gpr_mean = np.zeros((bins, bins, bins, 2))
		gpr_sigma = np.zeros((bins, bins, bins, 2, 2))

		quLATi_mean = np.zeros((bins, bins, bins, 2))
		quLATi_sigma = np.zeros((bins, bins, bins, 2, 2))

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

							pxls = np.column_stack(np.where((figEstquLATi[:, :, 0] >= r_v) & (figEstquLATi[:, :, 0] < binEdges[r_i + 1]) & \
										(figEstquLATi[:, :, 1] >= g_v) & (figEstquLATi[:, :, 1] < binEdges[g_i + 1]) & \
										(figEstquLATi[:, :, 2] >= b_v) & (figEstquLATi[:, :, 2] < binEdges[b_i + 1])))
							if pxls.shape[0] == 0:
								quLATi_sigma[r_i, g_i, b_i] = np.array([[1, 0], [0, 1]])
							else:
								quLATi_mean[r_i, g_i, b_i] = np.mean(pxls, axis=0)
								quLATi_sigma[r_i, g_i, b_i] = np.cov(pxls, rowvar=0)

		magic_corr.append(cv2.compareHist(true_hist_flat, magic_hist_flat, cv2.HISTCMP_CORREL))
		gpr_corr.append(cv2.compareHist(true_hist_flat, gpr_hist_flat, cv2.HISTCMP_CORREL))
		quLATi_corr.append(cv2.compareHist(true_hist_flat, quLATi_hist_flat, cv2.HISTCMP_CORREL))

		true_hist = cv2.normalize(true_hist, true_hist)
		magic_hist = cv2.normalize(magic_hist, magic_hist)
		gpr_hist = cv2.normalize(gpr_hist, gpr_hist)
		quLATi_hist = cv2.normalize(quLATi_hist, quLATi_hist)

		magic_similarity = 0
		gpr_similarity = 0
		quLATi_similarity = 0

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

					quLATi_js = js(true_mean[i,j,k], quLATi_mean[i,j,k], true_sigma[i,j,k], quLATi_sigma[i,j,k], true_hist[i,j,k], quLATi_hist[i,j,k])

					num = (true_hist[i,j,k] - np.mean(true_hist)) * (quLATi_hist[i,j,k] - np.mean(quLATi_hist))
					denom = math.sqrt(np.sum((true_hist - np.mean(true_hist)) ** 2) * np.sum((quLATi_hist - np.mean(quLATi_hist)) ** 2))
					quLATi_similarity += (num / denom) * math.exp(-1 * quLATi_js)

		magic_spatio_corr.append(magic_similarity)
		gpr_spatio_corr.append(gpr_similarity)
		quLATi_spatio_corr.append(quLATi_similarity)

	magicNMSE[test] = nmse
	magicMAE[test] = mae
	magicHistCorr[test] = np.mean(magic_corr)
	magicSpatioCorr[test] = np.mean(magic_spatio_corr)

	gprNMSE[test] = nmseGPR
	gprMAE[test] = maeGPR
	gprHistCorr[test] = np.mean(gpr_corr)
	gprSpatioCorr[test] = np.mean(gpr_spatio_corr)

	quLATiNMSE[test] = nmsequLATi
	quLATiMAE[test] = maequLATi
	quLATiHistCorr[test] = np.mean(quLATi_corr)
	quLATiSpatioCorr[test] = np.mean(quLATi_spatio_corr)


filename = os.path.join(outDir, 'p{}_t{:g}_m{:g}_tests{:g}.txt'.format(patient, EDGE_THRESHOLD, NUM_TRAIN_SAMPS, NUM_TEST_REPEATS))
with open(filename, 'w') as fid:
	fid.write('{:<30}{}\n'.format('file', nm))
	fid.write('{:<30}{:g}\n'.format('n', n))
	fid.write('{:<30}{:g}\n'.format('ignored', numPtsIgnored))
	fid.write('{:<30}{:g}/{:g}\n\n'.format('m', NUM_TRAIN_SAMPS, M))

	fid.write('{:<30}{:g}\n'.format('EDGE_THRESHOLD', EDGE_THRESHOLD))
	fid.write('{:<30}{:g}\n'.format('NUM_TEST_REPEATS', NUM_TEST_REPEATS))

	fid.write('\n\n')

	fid.write('MAGIC-LAT Performance\n\n')
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('NMSE', 	np.average(magicNMSE), 	np.std(magicNMSE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('MAE', 	np.average(magicMAE), 	np.std(magicMAE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('Histogram Corr.', 	np.average(magicHistCorr), 	np.std(magicHistCorr)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('Spatiogram Corr.', 	np.average(magicSpatioCorr), np.std(magicSpatioCorr)))

	fid.write('\n\n')

	fid.write('GPR Performance\n\n')
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('NMSE', 	np.average(gprNMSE), 	np.std(gprNMSE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('MAE', 	np.average(gprMAE), 	np.std(gprMAE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('Histogram Corr.', 	np.average(gprHistCorr), 	np.std(gprHistCorr)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('Spatiogram Corr.', 	np.average(gprSpatioCorr), 	np.std(gprSpatioCorr)))

	fid.write('\n\n')

	fid.write('quLATi Performance\n\n')
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('NMSE', 	np.average(quLATiNMSE), 	np.std(quLATiNMSE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('MAE', 	np.average(quLATiMAE), 	np.std(quLATiMAE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('Histogram Corr.', 	np.average(quLATiHistCorr), 	np.std(quLATiHistCorr)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('Spatiogram Corr.', 	np.average(quLATiSpatioCorr), 	np.std(quLATiSpatioCorr)))