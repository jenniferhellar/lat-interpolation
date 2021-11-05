
"""

DATA INDICES:
	Too large for my laptop:
		p031 = 0 (4-SINUS LVFAM)
		p032 = 1 (1-LVFAM LAT HYB), 2 (2-LVFAM INITIAL PVC), 3 (4-LVFAM SINUS)
		p037 = 10 (12-LV-SINUS)

	Testable:
		p033 = 4 (3-RV-FAM-PVC-A-NORMAL), 5 (4-RV-FAM-PVC-A-LAT-HYBRID)
		p034 = 6 (4-RVFAM-LAT-HYBRID), 7 (5-RVFAM-PVC), 8 (6-RVFAM-SINUS-VOLTAGE)
		p035 = 9 (8-SINUS)
		p037 = 11 (9-RV-SINUS-VOLTAGE)

Requirements: numpy, scipy, matplotlib, scikit-learn
"""

import os

import argparse

import numpy as np
import math
import random

from vedo import Plotter, Video, Points, Mesh
from vedo.pyplot import plot

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


import utils
import metrics
from const import DATADIR, DATAFILES
from magicLAT import magicLAT



EDGE_THRESHOLD			=		50

OUTDIR				 	=		'test_video_results'

""" Parse the input for data index argument. """
parser = argparse.ArgumentParser(
    description='Processes a single mesh file for comparison of MAGIC-LAT, GPR, and quLATi performance.')

parser.add_argument('-i', '--idx', required=True, default='11',
                    help='Data index to process. \
                    Default: 11')

parser.add_argument('-a', '--anomalies_removed', required=True, default=1,
                    help='Remove anomalous points (disable: 0, enable: 1). \
                    Default: 1')

parser.add_argument('-v', '--verbose', required=False, default=1,
                    help='Verbose output (disable: 0, enable: 1). \
                    Default: 1')

args = parser.parse_args()

PATIENT_IDX				=		int(vars(args)['idx'])
verbose					=		int(vars(args)['verbose'])
remove_anomalies		=		int(vars(args)['anomalies_removed'])

""" Obtain file names, patient number, mesh id, etc. """
(meshFile, latFile, ablFile) = DATAFILES[PATIENT_IDX]
nm = meshFile[0:-5]
patient = nm[7:10]
id = latFile.split('_')[3]

""" Create output directory for this script and subdir for this mesh. """
outSubDir = os.path.join(OUTDIR, 'p' + patient + '_' + id)
if not os.path.isdir(OUTDIR):
	os.makedirs(OUTDIR)
if not os.path.isdir(outSubDir):
	os.makedirs(outSubDir)

""" Read the files """
print('\nProcessing ' + nm + ' ...\n')
[vertices, faces] = readMesh(os.path.join(DATADIR, meshFile))
[OrigLatCoords, OrigLatVals] = readLAT(os.path.join(DATADIR, latFile))

if ablFile != '':
	ablFile = os.path.join(DATADIR, ablFile)
else:
	ablFile = None
	print('No ablation file available for this mesh... continuing...\n')

""" Pre-process the mesh and LAT samples. """
mesh = Mesh([vertices, faces])
mesh.c('grey')

n = len(vertices)

mapIdx = [i for i in range(n)]
mapCoord = [vertices[i] for i in mapIdx]

# Map the LAT samples to nearest mesh vertices
allLatIdx, allLatCoord, allLatVal = utils.mapSamps(mapIdx, mapCoord, OrigLatCoords, OrigLatVals)

M = len(allLatIdx)

# Identify and exclude anomalous LAT samples
if remove_anomalies:
	anomalous = utils.isAnomalous(allLatCoord, allLatVal)
else:
	anomalous = [0 for i in range(M)]

numPtsIgnored = np.sum(anomalous)

latIdx = [allLatIdx[i] for i in range(M) if anomalous[i] == 0]
latCoords = [allLatCoord[i] for i in range(M) if anomalous[i] == 0]
latVals = [allLatVal[i] for i in range(M) if anomalous[i] == 0]

M = len(latIdx)

# For colorbar ranges
MINLAT = math.floor(min(latVals)/10)*10
MAXLAT = math.ceil(max(latVals)/10)*10

# Create partially-sampled signal vector
mapLAT = [0 for i in range(n)]
for i in range(M):
	mapLAT[latIdx[i]] = latVals[i]


infoFile = os.path.join(outSubDir, 'p{}_info.txt'.format(patient))
meanFile = os.path.join(outSubDir, 'p{}_dE.txt'.format(patient))

with open(infoFile, 'w') as fid:
	fid.write('{:<30}{}\n'.format('file', nm))
	fid.write('{:<30}{:g}\n'.format('n', n))
	fid.write('{:<30}{:g}\n'.format('M', M))
	fid.write('{:<30}{:g}\n'.format('ignored', np.sum(anomalous)))

	fid.write('{:<30}{:g}\n'.format('EDGE_THRESHOLD', EDGE_THRESHOLD))

with open(meanFile, 'w') as fid:
	fid.write('{:<20}{:<20}'.format('m', 'dE'))


videoFile0 = os.path.join(outSubDir, 'p{}_front.mp4'.format(patient))
video0 = Video(videoFile0, fps=4, backend='opencv')

videoFile1 = os.path.join(outSubDir, 'p{}_back.mp4'.format(patient))
video1 = Video(videoFile1, fps=4, backend='opencv')

videoFile2 = os.path.join(outSubDir, 'p{}_dE.mp4'.format(patient))
video2 = Video(videoFile2, fps=4, backend='opencv')


# patient 033
# cam0 = {'pos': (-157, 128, 123),
#            'focalPoint': (14.1, 75.8, 115),
#            'viewup': (0.0728, 0.0926, 0.993),
#            'distance': 179,
#            'clippingRange': (110, 267)}
# cam1 = {'pos': (181, 13.0, 130),
#            'focalPoint': (14.1, 75.8, 115),
#            'viewup': (-0.0213, 0.172, 0.985),
#            'distance': 179,
#            'clippingRange': (106, 273)}

# patient 037
cam0 = {'pos': (183, 166, 4.95),
           'focalPoint': (-0.954, 31.3, 163),
           'viewup': (0.333, -0.873, -0.356),
           'distance': 277,
           'clippingRange': (136, 456)}

cam1 = {'pos': (-157, -63.4, 372),
           'focalPoint': (-0.954, 31.3, 163),
           'viewup': (0.154, -0.938, -0.310),
           'distance': 277,
           'clippingRange': (145, 444)}

x = []
y = []
maxDE = 0

tr_i = []
sampLst = utils.getModifiedSampList(latVals)

# for m in range(1, M-1):
tr_i = random.sample(sampLst, 25)
sampLst = [i for i in sampLst if i not in tr_i]	# to prevent repeats
stop = 250
for m in range(25, stop):

	print('adding sample #{:g} of {:g}...'.format(m, stop))
	
	elem = random.sample(sampLst, 1)[0]
	tr_i.append(elem)
	sampLst = [i for i in sampLst if i != elem]	# to prevent repeats
	tst_i = [i for i in range(M) if i not in tr_i]

	# tr_i = [i for i in range(m)]
	# tst_i = [i for i in range(m, M)]

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

	magicDE = metrics.deltaE(TstVal, latEst[TstIdx], MINLAT, MAXLAT)
	x.append(m)
	y.append(magicDE)
	if magicDE > maxDE:
		maxDE = magicDE
		ymax = math.ceil(magicDE/10)*10

	with open(meanFile, 'a') as fid:
		fid.write('\n')
		fid.write('{:<20}{:<20.6f}'.format(m, magicDE))

	verPoints = Points(TrCoord, r=5).cmap('rainbow_r', TrVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
	estPoints = Points(vertices, r=5).cmap('rainbow_r', latEst, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
	coloredMesh = Mesh([vertices, faces])
	coloredMesh.interpolateDataFrom(estPoints, N=1).cmap('rainbow_r', vmin=MINLAT, vmax=MAXLAT).addScalarBar()

	# dEPoints = Points(np.array(dE), r=5).c('black')

	vplt0 = Plotter(N=1, bg='black', resetcam=True, sharecam=False, offscreen=True)
	vplt0.show(coloredMesh, verPoints, title='Patient{}, Front View'.format(patient), camera=cam0)
	video0.addFrame()

	vplt1 = Plotter(N=1, bg='black', resetcam=True, sharecam=False, offscreen=True)
	vplt1.show(coloredMesh, verPoints, title='Patient{}, Back View'.format(patient), camera=cam1)
	video1.addFrame()

	vplt2 = Plotter(offscreen=True)
	errplt = plot(x, y, 'o', title='Patient{}, \DeltaE(m)'.format(patient), 
		xtitle='m', ytitle=r'\DeltaE', xlim=[0,m+1], ylim=[0,ymax], axes={'xyPlaneColor':'white'})
	vplt2.show(errplt)
	video2.addFrame()

	if m!= stop-1:
		vplt0.close()
		vplt1.close()
		vplt2.close()


video0.close()
video1.close()
video2.close()

