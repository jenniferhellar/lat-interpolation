
"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""

import os

# plotting packages
from vedo import *

# loading images
import cv2

from utils import *
import metrics


import math
import numpy as np

from matplotlib import pyplot as plt

import colour

def calcVisualMetrics(outDir, trueFigName, newFigName, idx=0):
	bins = 16
	binEdges = [i*256/bins for i in range(bins+1)]

	img = cv2.imread(os.path.join(outDir, trueFigName))
	dimX = img.shape[0]
	dimY = img.shape[1]
	n_black_px = np.sum(np.sum(img == 0, axis = 2) == 3)

	trueFig = cv2.imread(os.path.join(outDir, trueFigName))
	# trueFig = cv2.cvtColor(trueFig, cv2.COLOR_BGR2RGB)
	trueFig = cv2.cvtColor(trueFig.astype("float32") / 255, cv2.COLOR_BGR2LAB )
	# trueFig = rgb2lab(trueFig)

	# trueHist = cv2.calcHist([trueFig], [0, 1, 2], None, [bins, bins, bins],
	# 	[0, 256, 0, 256, 0, 256])
	trueHist = cv2.calcHist([trueFig],[0, 1, 2],None,[100, 64, 64],[0, 100, -128, 127, -128, 127])
	# trueHist[0, 0, 0] -= n_black_px
	# print(np.where(trueHist == max(trueHist.flatten())))
	trueHist[0, 32, 32] = 0

	newFig = cv2.imread(os.path.join(outDir, newFigName))
	# newFig = cv2.cvtColor(newFig, cv2.COLOR_BGR2RGB)
	newFig = cv2.cvtColor(newFig.astype("float32") / 255, cv2.COLOR_BGR2LAB )
	# newFig = rgb2lab(newFig)

	delta_E = colour.delta_E(trueFig, newFig, method='CIE 2000')
	# print(np.mean(delta_E))

	# newHist = cv2.calcHist([newFig], [0, 1, 2], None, [bins, bins, bins],
	# 	[0, 256, 0, 256, 0, 256])
	newHist = cv2.calcHist([newFig],[0, 1, 2],None,[100, 64, 64],[0, 100, -128, 127, -128, 127])
	# newHist[0, 0, 0] -= n_black_px
	newHist[0, 32, 32] = 0

	cv2.normalize(trueHist, trueHist)	
	cv2.normalize(newHist, newHist)

	# trueMean = np.zeros((bins, bins, bins, 2))
	# trueSigma = np.zeros((bins, bins, bins, 2, 2))

	# newMean = np.zeros((bins, bins, bins, 2))
	# newSigma = np.zeros((bins, bins, bins, 2, 2))

	trueMean = np.zeros((100, 64, 64, 2))
	trueSigma = np.zeros((100, 64, 64, 2, 2))

	newMean = np.zeros((100, 64, 64, 2))
	newSigma = np.zeros((100, 64, 64, 2, 2))

	L_edges = [i for i in range(100+1)]
	ab_edges = [i*256/64 - 128 for i in range(64+1)]

	# for r_i, r_v in enumerate(binEdges):
	# 	for g_i, g_v in enumerate(binEdges):
	# 		for b_i, b_v in enumerate(binEdges):
	for r_i, r_v in enumerate(L_edges):
		for g_i, g_v in enumerate(ab_edges):
			for b_i, b_v in enumerate(ab_edges):
				if r_v < 100 and g_v < 128 and b_v < 128:
					if trueHist[r_i, g_i, b_i] > 0 or newHist[r_i, g_i, b_i] > 0:
						# true_pxls = np.column_stack(np.where((trueFig[:, :, 0] >= r_v) & (trueFig[:, :, 0] < binEdges[r_i + 1]) & \
						# 	(trueFig[:, :, 1] >= g_v) & (trueFig[:, :, 1] < binEdges[g_i + 1]) & \
						# 	(trueFig[:, :, 2] >= b_v) & (trueFig[:, :, 2] < binEdges[b_i + 1])))
						true_pxls = np.column_stack(np.where((trueFig[:, :, 0] >= r_v) & (trueFig[:, :, 0] < L_edges[r_i + 1]) & \
							(trueFig[:, :, 1] >= g_v) & (trueFig[:, :, 1] < ab_edges[g_i + 1]) & \
							(trueFig[:, :, 2] >= b_v) & (trueFig[:, :, 2] < ab_edges[b_i + 1])))
						if true_pxls.shape[0] == 0:
							trueMean[r_i, g_i, b_i] = np.array([[dimX/2, dimY/2]])
							trueSigma[r_i, g_i, b_i] = np.array([[1, 0], [0, 1]])
						else:
							trueMean[r_i, g_i, b_i] = np.mean(true_pxls, axis=0)
							trueSigma[r_i, g_i, b_i] = np.cov(true_pxls, rowvar=0)
				
						# pxls = np.column_stack(np.where((newFig[:, :, 0] >= r_v) & (newFig[:, :, 0] < binEdges[r_i + 1]) & \
						# 			(newFig[:, :, 1] >= g_v) & (newFig[:, :, 1] < binEdges[g_i + 1]) & \
						# 			(newFig[:, :, 2] >= b_v) & (newFig[:, :, 2] < binEdges[b_i + 1])))
						pxls = np.column_stack(np.where((newFig[:, :, 0] >= r_v) & (newFig[:, :, 0] < L_edges[r_i + 1]) & \
									(newFig[:, :, 1] >= g_v) & (newFig[:, :, 1] < ab_edges[g_i + 1]) & \
									(newFig[:, :, 2] >= b_v) & (newFig[:, :, 2] < ab_edges[b_i + 1])))
						if pxls.shape[0] == 0:
							newMean[r_i, g_i, b_i] = np.array([[dimX/2, dimY/2]])
							newSigma[r_i, g_i, b_i] = np.array([[1, 0], [0, 1]])
						else:
							newMean[r_i, g_i, b_i] = np.mean(pxls, axis=0)
							newSigma[r_i, g_i, b_i] = np.cov(pxls, rowvar=0)

	plt.subplot(421, title='Ground truth image'), plt.imshow(trueFig)
	plt.subplot(422, title='Ground truth color histograms'),
	plt.plot(np.sum(np.sum(trueHist, axis=1), axis=1), 'r')
	plt.plot(np.sum(trueHist, axis=(0,2)), 'g')
	plt.plot(np.sum(np.sum(trueHist, axis=0), axis=0), 'b')
	plt.xlim([0,bins])
	plt.subplot(423, title='Test image'), plt.imshow(newFig)
	plt.subplot(424, title='Test color histograms'),
	plt.plot(np.sum(np.sum(newHist, axis=1), axis=1), 'r')
	plt.plot(np.sum(np.sum(newHist, axis=0), axis=1), 'g')
	plt.plot(np.sum(np.sum(newHist, axis=0), axis=0), 'b')
	plt.xlim([0,bins])

	plt.tight_layout()
	plt.savefig(os.path.join(outDir, 'colorHist{:g}.png'.format(idx)))
	# plt.show()

	plt.subplot(421, title='Ground truth image'), plt.imshow(trueFig)
	plt.subplot(422, title='Ground truth flattened histogram'),
	plt.plot(trueHist.flatten())
	plt.xlim([0,trueHist.flatten().shape[0]])
	plt.subplot(423, title='Test image'), plt.imshow(newFig)
	plt.subplot(424, title='Test flattened histogram'),
	plt.plot(newHist.flatten())
	plt.xlim([0,trueHist.flatten().shape[0]])

	plt.tight_layout()
	plt.savefig(os.path.join(outDir, 'hist{:g}.png'.format(idx)))

	corr = cv2.compareHist(trueHist.flatten(), newHist.flatten(), cv2.HISTCMP_CORREL)

	spatio_corr = 0

	# for i in range(bins):
	# 	for j in range(bins):
	# 		for k in range(bins):
	for i in range(100):
		for j in range(64):
			for k in range(64):
				if trueHist[i,j,k] > 0 or newHist[i,j,k] > 0:
					jsDist = metrics.js(trueMean[i,j,k], newMean[i,j,k], trueSigma[i,j,k], newSigma[i,j,k], trueHist[i,j,k], newHist[i,j,k])

					# Histogram correlation
					num = trueHist[i,j,k] * newHist[i,j,k]
					denom = math.sqrt(np.sum(trueHist ** 2) * np.sum(newHist ** 2))
					spatio_corr += (num / denom) * math.exp(-1 * jsDist)

	return (corr, spatio_corr)


outDir				 	=		'test_toy_results'
if not os.path.isdir(outDir):
	os.makedirs(outDir)

gridX = 10
gridY = 10
mesh = Grid(pos=(0,0,0), sx=gridX, sy=gridY, c='black')

vertices = mesh.vertices()
faces = mesh.faces()

mesh = Mesh([vertices, faces], c='black')

trueVals = []

for [x,y,z] in vertices:
	if x < 0:
		trueVals.append(-100)
	else:
		trueVals.append(100)

# mesh.backColor('violet').lineColor('tomato').lineWidth(2)

vplt = Plotter(N=1, axes=0, offscreen=True)
truePts = Points(vertices, r = 20).cmap('rainbow_r', trueVals, vmin=-100, vmax=100)
vplt.show(mesh, truePts, bg='black')
vplt.screenshot(filename=os.path.join(outDir, 'true.png'), returnNumpy=False)
vplt.close()

trueNMSE = metrics.calcNMSE(trueVals, trueVals)
trueMAE = metrics.calcMAE(trueVals, trueVals)
trueDE = metrics.deltaE(trueVals, trueVals, -100, 100)
# (trueCorr, trueSpatio) = calcVisualMetrics(outDir, 'true.png', 'true.png')

greenLineVals = []
for [x,y,z] in vertices:
	if x < 0:
		greenLineVals.append(-100)
	elif x == 0.0:
		greenLineVals.append(0)
	else:
		greenLineVals.append(100)


vplt = Plotter(N=1, axes=0, offscreen=True)
newGreenPts = Points(vertices, r = 20).cmap('rainbow_r', greenLineVals, vmin=-100, vmax=100)
vplt.show(mesh, newGreenPts, bg='black')
vplt.screenshot(filename=os.path.join(outDir, 'greenLine.png'), returnNumpy=False)
vplt.close()

greenLineNMSE = metrics.calcNMSE(trueVals, greenLineVals)
greenLineMAE = metrics.calcMAE(trueVals, greenLineVals)
greenLineDE = metrics.deltaE(trueVals, greenLineVals, -100, 100)
# (greenLineCorr, greenLineSpatio) = calcVisualMetrics(outDir, 'true.png', 'greenLine.png', idx=1)


shiftLineVals = []
for [x,y,z] in vertices:
	if x < 1:
		shiftLineVals.append(-100)
	else:
		shiftLineVals.append(100)


vplt = Plotter(N=1, axes=0, offscreen=True)
shiftLinePts = Points(vertices, r = 20).cmap('rainbow_r', shiftLineVals, vmin=-100, vmax=100)
vplt.show(mesh, shiftLinePts, bg='black')
vplt.screenshot(filename=os.path.join(outDir, 'shiftLine.png'), returnNumpy=False)
vplt.close()

shiftLineNMSE = metrics.calcNMSE(trueVals, shiftLineVals)
shiftLineMAE = metrics.calcMAE(trueVals, shiftLineVals)
shiftLineDE = metrics.deltaE(trueVals, shiftLineVals, -100, 100)
# (shiftLineCorr, shiftLineSpatio) = calcVisualMetrics(outDir, 'true.png', 'shiftLine.png', idx=2)

print(trueDE, greenLineDE, shiftLineDE)

with open(os.path.join(outDir, 'metrics.txt'), 'w') as fid:
	fid.write('{:>65}\n'.format('Metric'))
	fid.write('{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}\n'.format('', '', 'NMSE', 'MAE', 'DeltaE', 'Histogram Corr.', 'Spatiogram Corr.'))
	fid.write('{:<10}{:<20}{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('Image', 'true.png', 
		trueNMSE, trueMAE, trueDE, trueCorr, trueSpatio))
	fid.write('{:<10}{:<20}{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('', 'greenLine.png', 
		greenLineNMSE, greenLineMAE, greenLineDE, greenLineCorr, greenLineSpatio))
	fid.write('{:<10}{:<20}{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('', 'shiftLine.png', 
		shiftLineNMSE, shiftLineMAE, shiftLineDE, shiftLineCorr, shiftLineSpatio))
