"""
--------------------------------------------------------------------------------
Performance metric functions.
--------------------------------------------------------------------------------

Description: LAT interpolation performance metrics.

Requirements: numpy, cv2, matplotlib, colour

File: metrics.py

Author: Jennifer Hellar
Email: jenniferhellar@gmail.com
--------------------------------------------------------------------------------
"""

import numpy as np

import cv2
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import colour


def deltaE(trueVals, estVals, MINLAT, MAXLAT, cmap=cm.viridis_r):
	norm = Normalize(vmin=MINLAT, vmax=MAXLAT)
	m = cm.ScalarMappable(norm=norm, cmap=cmap)

	trueVals = list(np.array(trueVals).flatten())
	estVals = list(np.array(estVals).flatten())

	trueColors = m.to_rgba(trueVals)[:,:3]	# multiply by 255 for RGB

	trueColors = np.array([trueColors])	# cvtColor req 2-dim array

	trueColors = cv2.cvtColor(trueColors.astype("float32"), cv2.COLOR_RGB2LAB)
	estColors = m.to_rgba(estVals)[:,:3]
	estColors = np.array([estColors])
	estColors = cv2.cvtColor(estColors.astype("float32"), cv2.COLOR_RGB2LAB)

	dE = np.mean(colour.delta_E(trueColors, estColors, method='CIE 2000'))

	return dE


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


def calcSNR(sig, sigEst):
	n = len(sig)
	err = [abs(sigEst[i] - sig[i]) for i in range(n)]
	err = np.array(err)

	sigKnown = [sig[i] for i in range(n)]
	sigPower = np.sum((np.array(sigKnown) - np.mean(sigKnown)) ** 2)

	snr = 20*np.log10(sigPower/np.sum(err ** 2))

	return snr