"""
--------------------------------------------------------------------------------
MAGIC-LAT implementation.
--------------------------------------------------------------------------------

Description: Implements MAGIC-LAT and associated sub-functions.

Requirements: numpy, math, robust_laplacian, scipy

File: magicLAT.py

Author: Jennifer Hellar
Email: jenniferhellar@gmail.com
--------------------------------------------------------------------------------
"""

import numpy as np
import math

# Robust cotan-based Laplacian for triangle mesh
import robust_laplacian

# nearest-neighbor interpolation
from scipy.interpolate import griddata

# KD-Tree for mapping to nearest point
from scipy.spatial import cKDTree


def updateFaces(V, F, latTiled, knownV, thresh):
	"""
	Removes faces if a triangle edge has a value delta > thresh and has
	values estimated from measured values <15cm away.

	V: array of vertex coordinates
	F: list of triangles in the mesh

	latTiled: estimated signal values for every vertex
	knownV: vertices for which signal value is measured (not estimated)
	thresh: threshold for edge (face) removal
	"""
	newF = []

	# KD Tree to find the nearest known mesh vertex
	coordKDtree = cKDTree(knownV)

	for tri in F:	# a triangle is a triplet of vertex indices

		v0 = int(tri[0])	# indices of vertices 0, 1, and 2 in triangle
		v1 = int(tri[1])
		v2 = int(tri[2])

		lat0 = latTiled[v0]	# lat values (tiled manifold)
		lat1 = latTiled[v1]
		lat2 = latTiled[v2]

		# find distance from nearest known point to vertex 0
		[d0, _] = coordKDtree.query(V[v0, :], k=2)
		if d0[0] > 0:
			d0 = d0[0]
		else:
			d0 = d0[1]	# first point found may be itself
		# find distance from nearest known point to vertex 1
		[d1, _] = coordKDtree.query(V[v1, :], k=2)
		if d1[0] > 0:
			d1 = d1[0]
		else:
			d1 = d1[1]	# first point found may be itself
		# find distance from nearest known point to vertex 2
		[d2, _] = coordKDtree.query(V[v2, :], k=2)
		if d2[0] > 0:
			d2 = d2[0]
		else:
			d2 = d2[1]	# first point found may be itself

		e_01 = (abs(lat0 - lat1) < thresh) or (d0 > 15) or (d1 > 15)
		e_12 = (abs(lat1 - lat2) < thresh) or (d1 > 15) or (d2 > 15)
		e_20 = (abs(lat2 - lat0) < thresh) or (d2 > 15) or (d0 > 15)

		if e_01 and e_12 and e_20:
			newF.append(tri)

	return np.array(newF, dtype='int')


def magicLAT(V, F, trIdx, trCoord, trLAT, edgeThreshold=50, alpha=1e-5, beta=1e-2):

	N = len(V)	# number of vertices in the graph
	M = len(trIdx)			# number of signal samples

	IDX = [i for i in range(N)]
	COORD = [V[i] for i in IDX]
	lat = np.zeros((N,1))

	trCoord = [V[i] for i in trIdx]

	# (short) lists of sampled vertices, coordinates, and  LAT values
	for i in range(M):
		verIdx = trIdx[i]
		lat[verIdx] = trLAT[i]

	# NN interpolation of unknown vertices
	latNN = [0 for i in range(N)]
	unknownCoord = [COORD[i] for i in range(N) if i not in trIdx]
	unknownCoord = griddata(np.array(trCoord), np.array(trLAT), np.array(unknownCoord), method='nearest')
	currIdx = 0
	for i in range(N):
		if i not in trIdx:
			latNN[i] = unknownCoord[currIdx]
			currIdx += 1
		else:
			latNN[i] = lat[i]

	faces = updateFaces(V, F, latNN, trCoord, edgeThreshold)

	L, _ = robust_laplacian.mesh_laplacian(V, faces)

	latEst = np.zeros((N,1))

	M_l = np.zeros((N,N))
	M_u = np.zeros((N,N))

	for i in range(N):
		if i in trIdx:
			M_l[i,i] = float(1)
		else:
			M_u[i,i] = float(1)


	T = np.linalg.inv(M_l + alpha*M_u + beta*L)

	latEst = np.matmul(T, lat)

	for i in range(N):
		if i in trIdx:
			latEst[i] = lat[i]

	return latEst
