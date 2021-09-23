
"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""

import os

import numpy as np
import math
import random

# plotting packages
from vedo import *

# Gaussian process regression interpolation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


from utils import *
from metrics import *
from const import *
from magicLAT import *

from quLATiHelper import *



"""
Too large for my computer:
p031 = 3
p032 = 6

Testable:
p033 = 9
p034 = 14
p035 = 18
p037 = 21
"""
PATIENT_MAP				=		12

NUM_TRAIN_SAMPS 		= 		100
EDGE_THRESHOLD			=		50

NUM_TEST_REPEATS 		= 		20

outDir				 	=		'test_cotan_repeated_results'

""" Read the files """
meshFile = meshNames[PATIENT_MAP]
latFile = latNames[PATIENT_MAP]
nm = meshFile[0:-5]
patient = nm[7:10]

print('Reading files for ' + nm + ' ...\n')
[vertices, faces] = readMesh(os.path.join(dataDir, meshFile))
[OrigLatCoords, OrigLatVals] = readLAT(os.path.join(dataDir, latFile))

ablFile = os.path.join(dataDir, ablNames[patient])
if not os.path.isfile(ablFile):
	ablFile = None
	print('No ablation location file available for this patient.\n')

n = len(vertices)

mapIdx = [i for i in range(n)]
mapCoord = [vertices[i] for i in mapIdx]

allLatIdx, allLatCoord, allLatVal = mapSamps(mapIdx, mapCoord, OrigLatCoords, OrigLatVals)

M = len(allLatIdx)

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
	[edges, TRI] = edgeMatrix(vertices, faces)

	print('Writing edge matrix to file...')
	with open(edgeFile, 'wb') as fid:
		np.save(fid, edges)
else:
	edges = np.load(edgeFile, allow_pickle=True)

if not os.path.isdir(outDir):
	os.makedirs(outDir)

""" Create GPR kernel and regressor """
gp_kernel = RBF(length_scale=0.01) + RBF(length_scale=0.1) + RBF(length_scale=1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)

"""
Create the quLATi model
"""
model = quLATiModel(patient, vertices, faces)

""" Sampling """
sampLst = getModifiedSampList(latVals)

magicNMSE = [0 for i in range(NUM_TEST_REPEATS)]
magicMAE = [0 for i in range(NUM_TEST_REPEATS)]
magicDE = [0 for i in range(NUM_TEST_REPEATS)]

gprNMSE = [0 for i in range(NUM_TEST_REPEATS)]
gprMAE = [0 for i in range(NUM_TEST_REPEATS)]
gprDE = [0 for i in range(NUM_TEST_REPEATS)]

quLATiNMSE = [0 for i in range(NUM_TEST_REPEATS)]
quLATiMAE = [0 for i in range(NUM_TEST_REPEATS)]
quLATiDE = [0 for i in range(NUM_TEST_REPEATS)]

cotanNMSE = [0 for i in range(NUM_TEST_REPEATS)]
cotanMAE = [0 for i in range(NUM_TEST_REPEATS)]
cotanDE = [0 for i in range(NUM_TEST_REPEATS)]

# For colorbar ranges
MINLAT = math.floor(min(latVals)/10)*10
MAXLAT = math.ceil(max(latVals)/10)*10

for test in range(NUM_TEST_REPEATS):
	
	print('test #{:g} of {:g}.'.format(test + 1, NUM_TEST_REPEATS))

	samps = sampLst

	tr_i = []
	for i in range(NUM_TRAIN_SAMPS):
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
	latEst = magicLATunweighted(vertices, faces, edges, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)

	latEstcotan = magicLAT(vertices, faces, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)

	""" GPR estimate """
	gpr.fit(TrCoord, TrVal)
	latEstGPR = gpr.predict(vertices, return_std=False)

	""" quLATi estimate """
	latEstquLATi = quLATi(TrIdx, TrVal, vertices, model)


	"""
	Error metrics
	"""

	nmse = calcNMSE(TstVal, latEst[TstIdx])
	nmseGPR = calcNMSE(TstVal, latEstGPR[TstIdx])
	nmsequLATi = calcNMSE(TstVal, latEstquLATi[TstIdx])

	nmseCotan = calcNMSE(TstVal, latEstcotan[TstIdx])

	mae = calcMAE(TstVal, latEst[TstIdx])
	maeGPR = calcMAE(TstVal, latEstGPR[TstIdx])
	maequLATi = calcMAE(TstVal, latEstquLATi[TstIdx])

	maeCotan = calcMAE(TstVal, latEstcotan[TstIdx])

	dE = deltaE(TstVal, latEst[TstIdx], MINLAT, MAXLAT)
	dEGPR = deltaE(TstVal, latEstGPR[TstIdx], MINLAT, MAXLAT)
	dEquLATi = deltaE(TstVal, latEstquLATi[TstIdx], MINLAT, MAXLAT)

	dEcotan = deltaE(TstVal, latEstcotan[TstIdx], MINLAT, MAXLAT)

	magicNMSE[test] = nmse
	magicMAE[test] = mae
	magicDE[test] = dE

	cotanNMSE[test] = nmseCotan
	cotanMAE[test] = maeCotan
	cotanDE[test] = dEcotan

	gprNMSE[test] = nmseGPR
	gprMAE[test] = maeGPR
	gprDE[test] = dEGPR

	quLATiNMSE[test] = nmsequLATi
	quLATiMAE[test] = maequLATi
	quLATiDE[test] = dEquLATi


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
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('DeltaE-2000', 	np.average(magicDE), np.std(magicDE)))

	fid.write('\n\n')

	fid.write('Cotan Performance\n\n')
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('NMSE', 	np.average(cotanNMSE), 	np.std(cotanNMSE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('MAE', 	np.average(cotanMAE), 	np.std(cotanMAE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('DeltaE-2000', 	np.average(cotanDE), np.std(cotanDE)))

	fid.write('\n\n')

	fid.write('GPR Performance\n\n')
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('NMSE', 	np.average(gprNMSE), 	np.std(gprNMSE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('MAE', 	np.average(gprMAE), 	np.std(gprMAE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('DeltaE-2000', 	np.average(gprDE), np.std(gprDE)))

	fid.write('\n\n')

	fid.write('quLATi Performance\n\n')
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('NMSE', 	np.average(quLATiNMSE), 	np.std(quLATiNMSE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('MAE', 	np.average(quLATiMAE), 	np.std(quLATiMAE)))
	fid.write('{:<30}{:.4f} +/- {:.4f}\n'.format('DeltaE-2000', 	np.average(quLATiDE), np.std(quLATiDE)))