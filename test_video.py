
"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""

import os

import numpy as np
import math
import random

from vedo import Plotter, Video, Points, Mesh
from vedo.pyplot import plot

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


# from utils import *
import utils
# from metrics import *
import metrics
from const import *
from magicLAT import magicLAT


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
PATIENT_MAP				=		9

EDGE_THRESHOLD			=		50

outDir				 	=		'test_video_results'

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
meanFile = os.path.join(outDir, 'p{}_dE.txt'.format(patient))

with open(infoFile, 'w') as fid:
	fid.write('{:<30}{}\n'.format('file', nm))
	fid.write('{:<30}{:g}\n'.format('n', n))
	fid.write('{:<30}{:g}\n'.format('M', M))
	fid.write('{:<30}{:g}\n'.format('ignored', np.sum(anomalous)))

	fid.write('{:<30}{:g}\n'.format('EDGE_THRESHOLD', EDGE_THRESHOLD))

with open(meanFile, 'w') as fid:
	fid.write('{:<20}{:<20}'.format('m', 'dE'))


videoFile0 = os.path.join(outDir, 'p{}_front.mp4'.format(patient))
video0 = Video(videoFile0, fps=4, backend='opencv')
cam0 = {'pos': (-157, 128, 123),
           'focalPoint': (14.1, 75.8, 115),
           'viewup': (0.0728, 0.0926, 0.993),
           'distance': 179,
           'clippingRange': (110, 267)}

videoFile1 = os.path.join(outDir, 'p{}_back.mp4'.format(patient))
video1 = Video(videoFile1, fps=4, backend='opencv')
cam1 = {'pos': (181, 13.0, 130),
           'focalPoint': (14.1, 75.8, 115),
           'viewup': (-0.0213, 0.172, 0.985),
           'distance': 179,
           'clippingRange': (106, 273)}

videoFile2 = os.path.join(outDir, 'p{}_dE.mp4'.format(patient))
video2 = Video(videoFile2, fps=4, backend='opencv')

# For colorbar ranges
MINLAT = math.floor(min(latVals)/10)*10
MAXLAT = math.ceil(max(latVals)/10)*10

x = []
y = []
maxDE = 0

tr_i = []
sampLst = utils.getModifiedSampList(latVals)

for m in range(1, M-1):

	print('adding sample #{:g} of {:g}...'.format(m, M))
	
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

	if m!= M-2:
		vplt0.close()
		vplt1.close()
		vplt2.close()


video0.close()
video1.close()
video2.close()

