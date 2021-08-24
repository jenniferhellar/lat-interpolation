"""
--------------------------------------------------------------------------------
Utility functions for MAGIC-LAT.
--------------------------------------------------------------------------------

Description: Utility functions to compute graph edges from triangle mesh and
corresponding adjacency matrix from the edges.

Requirements: numpy

File: utils.py

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""

import numpy as np
import math

import os
from vedo import *

from scipy.spatial import cKDTree

import matplotlib.pyplot as plt



def mapSamps(IDX, COORD, coords, vals):
	"""
	Maps LAT sample values (vals) at coordinates (coords) not
	on the mesh to the nearest mesh coordinate in COORD.

	Returns:
	- latIdx, a list of mesh vertex indices with an LAT sample
	- latCoords, a list of corr. LAT vertex coordinates
	- latVals, a list of corr. LAT values

	"""
	n = len(IDX)	# number of vertices in the graph
	m = len(coords)			# number of signal samples

	# KD Tree to find the nearest mesh vertex
	coordKDtree = cKDTree(COORD)
	[_, nearestVer] = coordKDtree.query(coords, k=1)

	# find the vertices with an assigned (known) sample
	known = [False for i in range(n)]
	lat = [0 for i in range(n)]
	for sampPt in range(m):
		verIdx = nearestVer[sampPt]	# nearest vertex
		known[verIdx] = True
		lat[verIdx] = vals[sampPt]	# assign the value

	latIdx = [IDX[i] for i in range(n) if known[i] is True]
	latCoords = [COORD[i] for i in range(n) if known[i] is True]
	latVals = [lat[i] for i in range(n) if known[i] is True]

	return latIdx, latCoords, latVals


def isAnomalous(allLatCoord, allLatVal, k=6, d=5, thresh=50):
	# KD Tree to find the nearest mesh vertex
	coordKDtree = cKDTree(allLatCoord)
	[dist, nearestVers] = coordKDtree.query(allLatCoord, k=k)

	M = len(allLatCoord)
	anomalous = np.zeros(M)

	for i in range(M):
		verCoord = allLatCoord[i]
		verVal = allLatVal[i]

		neighbors = [nearestVers[i, n] for n in range(1,k) if dist[i,n] < d]

		adj = len(neighbors)

		cnt = 0
		for neighVer in neighbors:
			neighVal = allLatVal[neighVer]

			if abs(verVal - neighVal) > thresh:
				cnt += 1
			else:
				break

		# if (cnt >= (len(neighbors)-1) and len(neighbors) > 1):	# differs from all but 1 neighbor by >50ms and has at least 2 neighbors w/in 5mm
		if cnt > 1 and adj > 1:
			anomalous[i] = 1
			# print(cnt, adj)

			# print(verVal, [allLatVal[neighVer] for neighVer in neighbors])
	return anomalous


def getModifiedSampList(latVals):
	M = len(latVals)

	sort_index = np.argsort(np.array(latVals))
	sortedLATVals = [latVals[i] for i in sort_index]
	# pos = [int(abs(sortedLATVals[i] - max(sortedLATVals))) for i in range(M)]
	pos = [int(sortedLATVals[i] + abs(min(sortedLATVals))) for i in range(M)]

	ratiodiff = [-0.0005*abs(pos[i] - np.average(pos)) ** 2 for i in range(M)]
	ratio = [int(ratiodiff[i] - min(ratiodiff) + 1) for i in range(M)]
	print(min(ratio), max(ratio))

	sampLst = []
	for i in range(M):
		reps = ratio[i]
		idx = sort_index[i]
		if reps == 0.0:
			sampLst.append(idx)
		for r in range(int(reps)):
			sampLst.append(idx)
	# print(sampLst.count(latVals.index(min(latVals))))

	# plt.plot(sortedLATVals, ratio/np.sum(ratio), 'o')
	# plt.xlabel('LAT Value (ms)')
	# plt.ylabel('Sampling Probability')
	# plt.show()

	# exit(0)

	return sampLst


def createTestPointImages(vertices, faces, MINLAT, MAXLAT, outDir,
	coords, trueVals, magicVals, gprVals=[], quLATiVals=[]):

	mesh = Mesh([vertices, faces], c='black')

	"""
	Ground truth (test points only)
	"""
	plotSaveTestPoints(mesh, coords, trueVals,
		MINLAT, MAXLAT, outDir, fileprefix='true')

	"""
	MAGIC-LAT estimate (test points only)
	"""
	plotSaveTestPoints(mesh, coords, magicVals,
		MINLAT, MAXLAT, outDir, fileprefix='magic')

	"""
	GPR estimate (test points only)
	"""
	if len(gprVals) > 0:
		plotSaveTestPoints(mesh, coords, gprVals,
			MINLAT, MAXLAT, outDir, fileprefix='gpr')

	"""
	quLATi estimate (test points only)
	"""
	if len(quLATiVals) > 0:
		plotSaveTestPoints(mesh, coords, quLATiVals,
			MINLAT, MAXLAT, outDir, fileprefix='quLATi')


def plotSaveTestPoints(mesh, TstCoord, TstVal,
	MINLAT, MAXLAT, outDir, fileprefix):

	# vplt = Plotter(N=1, axes=0, offscreen=True)
	testPoints = Points(TstCoord, r=20).cmap('rainbow_r', TstVal, vmin=MINLAT, vmax=MAXLAT)

	elev = 0
	roll = 0
	azim = [0, 90, 180, 270]
	for a in azim:
		vplt = Plotter(N=1, axes=0, offscreen=True)
		vplt.show(mesh, testPoints, azimuth=a, elevation=elev, roll=roll, bg='black')
		vplt.screenshot(filename=os.path.join(outDir, fileprefix+'_elev{:g}azim{:g}.png'.format(elev, a)), returnNumpy=False)
		vplt.close()

	elev = [-90, 90]
	roll = 0
	azim = 0
	for e in elev:
		vplt = Plotter(N=1, axes=0, offscreen=True)
		vplt.show(mesh, testPoints, azimuth=azim, elevation=e, roll=roll, bg='black')
		vplt.screenshot(filename=os.path.join(outDir, fileprefix+'_elev{:g}azim{:g}.png'.format(e, azim)), returnNumpy=False)
		vplt.close()


def plotSaveEntire(mesh, latCoords, latVals, TrCoord, TrVal, latEst, 
	azimuth, elev, roll, MINLAT, MAXLAT, outDir, title, filename, ablFile=None):

	vertices = mesh.points()
	faces = mesh.faces()

	verPoints = Points(latCoords, r=5).cmap('rainbow_r', latVals, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
	largeVerPoints = Points(latCoords, r=10).cmap('rainbow_r', latVals, vmin=MINLAT, vmax=MAXLAT).addScalarBar()

	vplt = Plotter(N=3, axes=9, offscreen=True)

	# Plot 0: Ground truth
	vplt.show(mesh, verPoints, 'all known points', azimuth=azimuth, elevation=elev, roll=roll, at=0)

	# Plot 1: Training points
	trainPoints = Points(TrCoord, r=5).cmap('rainbow_r', TrVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
	vplt.show(mesh, trainPoints, 'training points', at=1)

	# Plot 2: MAGIC-LAT output signal
	estPoints = Points(vertices, r=5).cmap('rainbow_r', latEst, vmin=MINLAT, vmax=MAXLAT).addScalarBar()

	coloredMesh = Mesh([vertices, faces])

	coloredMesh.interpolateDataFrom(estPoints, N=1).cmap('rainbow_r', vmin=MINLAT, vmax=MAXLAT).addScalarBar()

	if ablFile != None:
		ablV = []
		with open(ablFile, 'r') as fID:
			for line in fID:
				lineSplit = line.split(' ')
				lineSplit = [i.strip() for i in lineSplit if i.strip() != '']
				x = float(lineSplit[0])
				y = float(lineSplit[1])
				z = float(lineSplit[2])
				ablV.append([x, y, z])
		ablMesh = Mesh([np.array(ablV), None], c='black')
		ptNormals = shapes.NormalLines(ablMesh, ratio=2, atCells=False, scale=2)
		vplt.show(coloredMesh, ptNormals, verPoints, 'interpolation result', title=title, at=2, interactive=True)
	else:
		vplt.show(coloredMesh, estPoints, verPoints, 'interpolation result', title=title, at=2, interactive=True)
	vplt.screenshot(filename=os.path.join(outDir, filename), returnNumpy=False)
	vplt.close()


def getPerspective(patient):
	if patient == '033':
		elev = 0
		azimuth = 90
		roll = 0
	elif patient == '034':
		elev = 0
		azimuth = 120
		roll = -45
	elif patient == '035':
		elev = 0
		azimuth = 0
		roll = 0
	elif (patient == '037'):
		elev = 0
		azimuth = 160
		roll = 0
	else:
		print('no specified plot view for this patient')
		elev = 0
		azimuth = 0
		roll = 0
	return elev, azimuth, roll