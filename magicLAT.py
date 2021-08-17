
import numpy as np

import math
# plotting packages
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from numpy.linalg import norm

import robust_laplacian

# nearest-neighbor interpolation
from scipy.interpolate import griddata


def edgeMatrix(coordinateMatrix, connectivityMatrix):
	"""
	Computes a list of edges in the graph, based on the triangles in
	connectivityMatrix, removing edges with a deltaLAT > thresh.  Returns
	a numpy array of the edges and a list of midpoints of removed edges,
	computed based on vertex coordinates in cooordinateMatrix.
	"""
	edges = []
	triangles = []

	for tri in connectivityMatrix:	# a triangle is a triplet of vertex indices

		idx0 = int(tri[0])	# indices of vertices 0, 1, and 2 in triangle
		idx1 = int(tri[1])
		idx2 = int(tri[2])

		# edges and corresponding deltaLATs
		e1 = set([idx0, idx1])
		e2 = set([idx1, idx2])
		e3 = set([idx0, idx2])

		# check if first edge seen before
		if (e1 not in edges):
			edges.append(e1)
			triangles.append([tri])
		else:
			k = edges.index(e1)
			triangles[k].append(tri)

		# repeat for second edge
		if (e2 not in edges):
			edges.append(e2)
			triangles.append([tri])
		else:
			k = edges.index(e2)
			triangles[k].append(tri)

		# repeat for third edge
		if (e3 not in edges):
			edges.append(e3)
			triangles.append([tri])
		else:
			k = edges.index(e3)
			triangles[k].append(tri)

	return [np.array(edges), triangles]



def updateEdges(coordinateMatrix, edges, lat, thresh):
	newEdges = []

	excl_midpt = []

	for i in range(len(edges)):
		e = list(edges[i])
		v_i = e[0]
		v_j = e[1]
		lat_i = lat[v_i]
		lat_j = lat[v_j]

		if abs(lat_j - lat_i) < thresh:
			newEdges.append(e)
		else:
			[x1, y1, z1] = coordinateMatrix[v_i, :]
			[x2, y2, z2] = coordinateMatrix[v_j, :]
			[x, y, z] = [float(x1+x2)/2, float(y1+y2)/2, float(z1+z2)/2]
			excl_midpt.append([x, y, z])

	return [np.array(newEdges), np.array(excl_midpt)]



def getUnWeightedAdj(n, edges):
	""" Computes the binary adjacency matrix """
	A = np.zeros((n, n))

	for i in range(len(edges)):

		e = list(edges[i])	# list of edges
		v_i = e[0]	# vertices given as indices
		v_j = e[1]

		A[v_i, v_j] = 1
		A[v_j, v_i] = 1

	return A



def magicLAT(V, F, E, trIdx, trCoord, trLAT, edgeThreshold=50, alpha=5e-05, beta=1):

	N = len(V)	# number of vertices in the graph
	M = len(trIdx)			# number of signal samples

	if N > 11000:
		print('Graph too large!')
		exit(0)

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

	[edges, excl_midpt] = updateEdges(V, E, latNN, edgeThreshold)

	A = getUnWeightedAdj(N, edges)

	D = np.diag(A.sum(axis=1))

	L = D - A

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



def magicLATcotan(V, F, E, trIdx, trCoord, trLAT, edgeThreshold=50, alpha=1e-5, beta=1):

	N = len(V)	# number of vertices in the graph
	M = len(trIdx)			# number of signal samples

	if N > 11000:
		print('Graph too large!')
		exit(0)

	IDX = [i for i in range(N)]
	COORD = [V[i] for i in IDX]
	lat = np.zeros((N,1))

	trCoord = [V[i] for i in trIdx]

	# (short) lists of sampled vertices, coordinates, and  LAT values
	for i in range(M):
		verIdx = trIdx[i]
		lat[verIdx] = trLAT[i]

	L, M = robust_laplacian.mesh_laplacian(V, np.array(F, dtype='int'))

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
