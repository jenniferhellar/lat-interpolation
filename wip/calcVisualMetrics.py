
import os
import cv2
import math
import numpy as np

from matplotlib import pyplot as plt

from js import *

# from skimage.color import rgb2lab

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
					jsDist = js(trueMean[i,j,k], newMean[i,j,k], trueSigma[i,j,k], newSigma[i,j,k], trueHist[i,j,k], newHist[i,j,k])

					# Histogram correlation
					num = trueHist[i,j,k] * newHist[i,j,k]
					denom = math.sqrt(np.sum(trueHist ** 2) * np.sum(newHist ** 2))
					spatio_corr += (num / denom) * math.exp(-1 * jsDist)

	return (corr, spatio_corr)