
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


# from utils import *
import utils
# from metrics import *
import metrics
from const import *
from magicLAT import magicLAT

# from quLATiHelper import *
import quLATiHelper



"""
To large for my computer:
p031 = 3
p032 = 6

Testable:
p033 = 9
p034 = 14
p035 = 18
p037 = 21
"""
PATIENT_MAP				=		14

EDGE_THRESHOLD			=		50

NUM_TEST_REPEATS 		= 		20

m						=		[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

outDir				 	=		'test_varied_m_results'

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

allLatIdx, allLatCoord, allLatVal = utils.mapSamps(mapIdx, mapCoord, OrigLatCoords, OrigLatVals)

M = len(allLatIdx)

anomalous = utils.isAnomalous(allLatCoord, allLatVal, k=6, d=5, thresh=50)

latIdx = [allLatIdx[i] for i in range(M) if anomalous[i] == 0]
latCoords = [allLatCoord[i] for i in range(M) if anomalous[i] == 0]
latVals = [allLatVal[i] for i in range(M) if anomalous[i] == 0]

M = len(latIdx)

mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]

if not os.path.isdir(outDir):
	os.makedirs(outDir)

infoFile = os.path.join(outDir, 'p{}_info.txt'.format(patient))
meanFile = os.path.join(outDir, 'p{}_mean.txt'.format(patient))
stdFile = os.path.join(outDir, 'p{}_std.txt'.format(patient))

with open(infoFile, 'w') as fid:
	fid.write('{:<30}{}\n'.format('file', nm))
	fid.write('{:<30}{:g}\n'.format('n', n))
	fid.write('{:<30}{:g}\n'.format('M', M))
	fid.write('{:<30}{:g}\n'.format('ignored', np.sum(anomalous)))

	fid.write('{:<30}{:g}\n'.format('EDGE_THRESHOLD', EDGE_THRESHOLD))
	fid.write('{:<30}{:g}\n'.format('NUM_TEST_REPEATS', NUM_TEST_REPEATS))

with open(meanFile, 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}'.format('m', 'MAGIC-LAT', 'GPR', 'quLATi'))

with open(stdFile, 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}{:<20}'.format('m', 'MAGIC-LAT', 'GPR', 'quLATi'))

""" Create GPR kernel and regressor """
gp_kernel = RBF(length_scale=0.01) + RBF(length_scale=0.1) + RBF(length_scale=1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)

"""
Create the quLATi model
"""
model = quLATiHelper.quLATiModel(patient, vertices, faces)

""" Sampling """
sampLst = utils.getModifiedSampList(latVals)

# For colorbar ranges
MINLAT = math.floor(min(latVals)/10)*10
MAXLAT = math.ceil(max(latVals)/10)*10

for i in range(len(m)):

	numSamps = m[i]

	print('testing #{:g} of {:g} possible m values.'.format(i + 1, len(m)))

	magicDE = [0 for i in range(NUM_TEST_REPEATS)]
	gprDE = [0 for i in range(NUM_TEST_REPEATS)]
	quLATiDE = [0 for i in range(NUM_TEST_REPEATS)]

	for test in range(NUM_TEST_REPEATS):
		
		print('\ttest #{:g} of {:g}.'.format(test + 1, NUM_TEST_REPEATS))

		samps = sampLst

		tr_i = []
		for i in range(numSamps):
			elem = random.sample(samps, 1)[0]
			tr_i.append(elem)
			samps = [i for i in samps if i != elem]	# to prevent repeats
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
		gpr.fit(TrCoord, TrVal)
		latEstGPR = gpr.predict(vertices, return_std=False)

		""" quLATi estimate """
		latEstquLATi = quLATiHelper.quLATi(TrIdx, TrVal, vertices, model)


		magicDE[test] = metrics.deltaE(TstVal, latEst[TstIdx], MINLAT, MAXLAT)
		gprDE[test] = metrics.deltaE(TstVal, latEstGPR[TstIdx], MINLAT, MAXLAT)
		quLATiDE[test] = metrics.deltaE(TstVal, latEstquLATi[TstIdx], MINLAT, MAXLAT)

	with open(meanFile, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}'.format(numSamps, np.average(magicDE), np.average(gprDE), np.average(quLATiDE)))

	with open(stdFile, 'a') as fid:
			fid.write('\n')
			fid.write('{:<20}{:<20.6f}{:<20.6f}{:<20.6f}'.format(numSamps, np.std(magicDE), np.std(gprDE), np.std(quLATiDE)))