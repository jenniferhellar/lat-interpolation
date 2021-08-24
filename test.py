
"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""

import os

import numpy as np
import math
import random

# Gaussian process regression interpolation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


from utils import *
from const import *
from metrics import *
from magicLAT import *

from quLATiHelper import *


"""
p033 = 9
p034 = 14
p035 = 18
p037 = 21
"""
PATIENT_MAP				=		9

NUM_TRAIN_SAMPS 		= 		100

EDGE_THRESHOLD			=		50

outDir				 	=		'test_results'

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

anomalous = isAnomalous(allLatCoord, allLatVal, k=6, d=5, thresh=50)

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
	[edges, triangles] = edgeMatrix(vertices, faces)

	print('Writing edge matrix to file...')
	with open(edgeFile, 'wb') as fid:
		np.save(fid, edges)
else:
	edges = np.load(edgeFile, allow_pickle=True)

if not os.path.isdir(outDir):
	os.makedirs(outDir)


""" Sampling """
sampLst = getModifiedSampList(latVals)

tr_i = []
for i in range(NUM_TRAIN_SAMPS):
	elem = random.sample(sampLst, 1)[0]
	tr_i.append(elem)
	sampLst = [i for i in sampLst if i != elem]	# to prevent repeats
tst_i = [i for i in range(M) if i not in tr_i]

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
latEst = magicLAT(vertices, faces, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)


""" GPR estimate """
gp_kernel = RBF(length_scale=0.01) + RBF(length_scale=0.1) + RBF(length_scale=1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)

# fit the GPR with training samples
gpr.fit(TrCoord, TrVal)

# predict the entire signal
latEstGPR = gpr.predict(vertices, return_std=False)


""" quLATi estimate """
model = quLATiModel(patient, vertices, faces)
latEstquLATi = quLATi(TrIdx, TrVal, vertices, model)



# For colorbar ranges
MINLAT = math.floor(min(allLatVal)/10)*10
MAXLAT = math.ceil(max(allLatVal)/10)*10

elev, azimuth, roll = getPerspective(patient)

"""
Figure 0: Ground truth (entire), training points, and MAGIC-LAT (entire)
"""
plotSaveEntire(mesh, latCoords, latVals, TrCoord, TrVal, latEst, 
	azimuth, elev, roll, MINLAT, MAXLAT,
	outDir, title='MAGIC-LAT', filename='magic.png')

"""
Figure 1: Ground truth (entire), training points, and GPR (entire)
"""
plotSaveEntire(mesh, latCoords, latVals, TrCoord, TrVal, latEstGPR, 
	azimuth, elev, roll, MINLAT, MAXLAT,
	outDir, title='GPR', filename='gpr.png')

"""
Figure 2: Ground truth (entire), training points, and quLATi (entire)
"""
plotSaveEntire(mesh, latCoords, latVals, TrCoord, TrVal, latEstquLATi, 
	azimuth, elev, roll, MINLAT, MAXLAT,
	outDir, title='quLATi', filename='quLATi.png')

# mesh.interpolateDataFrom(pts, N=1).cmap('rainbow_r').addScalarBar()

# createTestPointImages(vertices, faces, MINLAT, MAXLAT, outDir, 
# 	TstCoord, TstVal, latEst[TstIdx], latEstGPR[TstIdx], latEstquLATi[TstIdx])


"""
Error metrics
"""

nmse = calcNMSE(TstVal, latEst[TstIdx])
nmseGPR = calcNMSE(TstVal, latEstGPR[TstIdx])
nmsequLATi = calcNMSE(TstVal, latEstquLATi[TstIdx])

mae = calcMAE(TstVal, latEst[TstIdx])
maeGPR = calcMAE(TstVal, latEstGPR[TstIdx])
maequLATi = calcMAE(TstVal, latEstquLATi[TstIdx])

dE = deltaE(TstVal, latEst[TstIdx], MINLAT, MAXLAT)
dEGPR = deltaE(TstVal, latEstGPR[TstIdx], MINLAT, MAXLAT)
dEquLATi = deltaE(TstVal, latEstquLATi[TstIdx], MINLAT, MAXLAT)

# magic_spatio_corr = []
# gpr_spatio_corr = []
# quLATi_spatio_corr = []

# magic_corr = []
# gpr_corr = []
# quLATi_corr = []

# bins = 16
# binEdges = [i*256/bins for i in range(bins+1)]

# elev = 0
# azim = [0, 90, 180, 270]

# for a in azim:
# 	magic_scorr, magic_cor, gpr_scorr, gpr_cor, quLATi_scorr, quLATi_cor = colorHistAndSpatioCorr(outDir, 'true_elev{:g}azim{:g}.png'.format(elev, a),
# 		'magic_elev{:g}azim{:g}.png'.format(elev, a),
# 		'gpr_elev{:g}azim{:g}.png'.format(elev, a),
# 		'quLATi_elev{:g}azim{:g}.png'.format(elev, a),
# 		bins, binEdges,
# 		'{:g}azim{:g}.png'.format(elev, a))

# 	magic_corr.append(magic_cor)
# 	gpr_corr.append(gpr_cor)
# 	quLATi_corr.append(quLATi_cor)

# 	magic_spatio_corr.append(magic_scorr)
# 	gpr_spatio_corr.append(gpr_scorr)
# 	quLATi_spatio_corr.append(quLATi_scorr)

# elev = [-90, 90]
# azim = 0

# for e in elev:
# 	magic_scorr, magic_cor, gpr_scorr, gpr_cor, quLATi_scorr, quLATi_cor = colorHistAndSpatioCorr(outDir, 'true_elev{:g}azim{:g}.png'.format(e, azim),
# 		'magic_elev{:g}azim{:g}.png'.format(e, azim),
# 		'gpr_elev{:g}azim{:g}.png'.format(e, azim),
# 		'quLATi_elev{:g}azim{:g}.png'.format(e, azim),
# 		bins, binEdges,
# 		'{:g}azim{:g}.png'.format(e, azim))

# 	magic_corr.append(magic_cor)
# 	gpr_corr.append(gpr_cor)
# 	quLATi_corr.append(quLATi_cor)

# 	magic_spatio_corr.append(magic_scorr)
# 	gpr_spatio_corr.append(gpr_scorr)
# 	quLATi_spatio_corr.append(quLATi_scorr)

with open(os.path.join(outDir, 'metrics.txt'), 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}\n\n'.format('Metric', 'MAGIC-LAT', 'GPR', 'quLATi'))
	fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('NMSE', nmse, nmseGPR, nmsequLATi))
	fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('MAE', mae, maeGPR, maequLATi))

	fid.write('\nColor-Based\n')
	# fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('Histogram Corr.', np.mean(magic_corr), np.mean(gpr_corr), np.mean(quLATi_corr)))
	# fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('Spatiogram Corr.', np.mean(magic_spatio_corr), np.mean(gpr_spatio_corr), np.mean(quLATi_spatio_corr)))
	fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('DeltaE-2000', dE, dEGPR, dEquLATi))