
# functions to read the files
from readMesh import readMesh
from readLAT import readLAT

import utils

import os

import numpy as np

parentFolder = 'D:\jhell\Work\code\dataExtraction\TestExport'

fileList = os.listdir(parentFolder)
carfileList = [x for x in fileList if '_car.txt' in x]
meshfileList = [x for x in fileList if '.mesh' in x]

print('{:<70}{:<20}{:<20}\n'.format('file', 'n', 'M'))
for idx in range(len(carfileList)):
	meshFile = meshfileList[idx]
	latFile = carfileList[idx]

	[vertices, faces] = readMesh(os.path.join(parentFolder, meshFile))
	[OrigLatCoords, OrigLatVals] = readLAT(os.path.join(parentFolder, latFile))

	n = len(vertices)

	mapIdx = [i for i in range(n)]
	mapCoord = [vertices[i] for i in mapIdx]

	if len(OrigLatVals) > 0:
		allLatIdx, allLatCoord, allLatVal = utils.mapSamps(mapIdx, mapCoord, OrigLatCoords, OrigLatVals)

		M = len(allLatIdx)
	else:
		M = 0

	print('{:<70}{:<20}{:<20}'.format(latFile, n, M))