
import os

import numpy as np

import cv2

from utils import *

GPR 	=		0
GRAPH	=		1

PATIENT	=		'034'

m 		=		300


fileRes = 'p{}_t50_m{:g}_resLat.txt'.format(PATIENT, m)
fileLat = 'p{}_t50_m{:g}_latVals.txt'.format(PATIENT, m)

fileFigEst = 'p{}_t50_m{:g}_testpts.png'.format(PATIENT, m)
fileFigTruth = 'p{}_t50_m{:g}_truth.png'.format(PATIENT, m)

if GPR:
	resdir = 'results_gpr'
	fileRes = 'gpr_' + fileRes
	fileLat = 'gpr_' + fileLat
	fileFigEst = 'gpr_' + fileFigEst
	fileFigTruth = 'gpr_' + fileFigTruth
elif GRAPH:
	resdir = 'results'
else:
	print('Select an interpolation method.')
	exit()

fileRes = os.path.join(resdir, fileRes)
fileLat = os.path.join(resdir, fileLat)

fileFigEst = os.path.join(resdir, fileFigEst)
fileFigTruth = os.path.join(resdir, fileFigTruth)

resLAT = np.fromfile(fileRes, dtype=float, count=-1, sep='\n')
latVals = np.fromfile(fileLat, dtype=float, count=-1, sep='\n')

nmse = calcNMSE(latVals, resLAT)
snr = calcSNR(latVals, resLAT)
perc = calcPercError(latVals, resLAT)
mae = calcMAE(latVals, resLAT)
nrmse = calcNRMSE(latVals, resLAT)

figEst = cv2.imread(fileFigEst)
figTruth = cv2.imread(fileFigTruth)

ssim = calcSSIM(figTruth, figEst)

print('{:<20}{:.4f}'.format('NMSE', nmse))
print('{:<20}{:.4f}'.format('SNR', snr))
print('{:<20}{:.4f}'.format('PERC', perc))
print('{:<20}{:.4f}'.format('MAE', mae))
print('{:<20}{:.4f}'.format('NRMSE', nrmse))
print('{:<20}{:.4f}'.format('SSIM', ssim))