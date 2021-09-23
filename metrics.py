import numpy as np

import cv2
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import colour

import matplotlib.pyplot as plt


def calcMSE(sig, sigEst):
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


def deltaE(trueVals, estVals, MINLAT, MAXLAT, cmap=cm.viridis):
	norm = Normalize(vmin=MINLAT, vmax=MAXLAT)
	m = cm.ScalarMappable(norm=norm, cmap=cmap)

	trueVals = list(np.array(trueVals).flatten())
	estVals = list(np.array(estVals).flatten())

	trueColors = m.to_rgba(trueVals)[:,:3]	# multiply by 255 for RGB

	trueColors = np.array([trueColors])	# cvtColor req 2-dim array
	# print(trueColors[0, 0])

	trueColors = cv2.cvtColor(trueColors.astype("float32"), cv2.COLOR_RGB2LAB)
	# print(trueColors[0, 0])
	estColors = m.to_rgba(estVals)[:,:3]
	estColors = np.array([estColors])
	estColors = cv2.cvtColor(estColors.astype("float32"), cv2.COLOR_RGB2LAB)

	dE = np.mean(colour.delta_E(trueColors, estColors, method='CIE 2000'))
	# dE = colour.delta_E(trueColors, estColors, method='CIE 2000')

	return dE


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