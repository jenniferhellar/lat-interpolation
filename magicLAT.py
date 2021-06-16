
import numpy as np

import math
# plotting packages
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# from cotanW import *

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


def getE(coordinateMatrix, connectivityMatrix, LAT, thresh):
	"""
	Computes a list of edges in the graph, based on the triangles in
	connectivityMatrix, removing edges with a deltaLAT > thresh.  Returns
	a numpy array of the edges and a list of midpoints of removed edges,
	computed based on vertex coordinates in cooordinateMatrix.
	"""
	edges = []
	triangles = []

	excl_edges = []
	excl_midpt = []
	for tri in connectivityMatrix:	# a triangle is a triplet of vertex indices

		idx0 = int(tri[0])	# indices of vertices 0, 1, and 2 in triangle
		idx1 = int(tri[1])
		idx2 = int(tri[2])

		# edges and corresponding deltaLATs
		e1 = set([idx0, idx1])
		e1_lat = abs(LAT[idx0] - LAT[idx1])
		e2 = set([idx1, idx2])
		e2_lat = abs(LAT[idx1] - LAT[idx2])
		e3 = set([idx0, idx2])
		e3_lat = abs(LAT[idx0] - LAT[idx2])

		# check if first edge seen before
		if (e1 not in edges):
			if (e1 not in excl_edges):
				# for new edge, check if it should be included
				if (e1_lat < thresh):
					edges.append(e1)
					triangles.append([tri])
				# if removed, compute midpoint
				else:
					[x1, y1, z1] = coordinateMatrix[idx0, :]
					[x2, y2, z2] = coordinateMatrix[idx1, :]
					[x, y, z] = [float(x1+x2)/2, float(y1+y2)/2, float(z1+z2)/2]
					excl_edges.append(e1)
					excl_midpt.append([x, y, z])
		else:
			k = edges.index(e1)
			triangles[k].append(tri)

		# repeat for second edge
		if (e2 not in edges):
			if (e2 not in excl_edges):
				if (e2_lat < thresh):
					edges.append(e2)
					triangles.append([tri])
				else:
					[x1, y1, z1] = coordinateMatrix[idx1, :]
					[x2, y2, z2] = coordinateMatrix[idx2, :]
					[x, y, z] = [float(x1+x2)/2, float(y1+y2)/2, float(z1+z2)/2]
					excl_edges.append(e2)
					excl_midpt.append([x, y, z])
		else:
			k = edges.index(e2)
			triangles[k].append(tri)

		# repeat for third edge
		if (e3 not in edges):
			if (e3 not in excl_edges):
				if (e3_lat < thresh):
					edges.append(e3)
					triangles.append([tri])
				else:
					[x1, y1, z1] = coordinateMatrix[idx0, :]
					[x2, y2, z2] = coordinateMatrix[idx2, :]
					[x, y, z] = [float(x1+x2)/2, float(y1+y2)/2, float(z1+z2)/2]
					excl_edges.append(e3)
					excl_midpt.append([x, y, z])
		else:
			k = edges.index(e3)
			triangles[k].append(tri)

	# for tri2 in triangles:
	# 	if len(tri2) != 2:
	# 		print(tri2)

	return [np.array(edges), triangles, np.array(excl_midpt)]


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


def magicLAT(V, F, E, trIdx, trCoord, trLAT, edgeThreshold=50, alpha=1e-03, beta=1):

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

	# [EDGES, TRI, excl_midpt] = getE(V, F, latNN, edgeThreshold)
	[edges, excl_midpt] = updateEdges(V, E, latNN, edgeThreshold)

	# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8), subplot_kw=dict(projection="3d"))

	# # For colorbar ranges
	# MINLAT = math.floor(min(trLAT)/10)*10
	# MAXLAT = math.ceil(max(trLAT)/10)*10

	# # Figure view
	# elev = 24
	# azim = -135

	# pltSig = trLAT
	# pltCoord = np.array(trCoord)

	# triang = mtri.Triangulation(V[:,0], V[:,1], triangles=F)
	# ax.plot_trisurf(triang, V[:,2], color=(0,0,0,0), edgecolor='k', linewidth=.1)
	# pos = ax.scatter(pltCoord[:,0], pltCoord[:,1], pltCoord[:,2], c=pltSig, cmap='rainbow_r', vmin=MINLAT, vmax=MAXLAT, s = 5)

	# # for i in range(S_N):
	# # 	txt = str(S_IDX[i])
	# # 	ax.text(coordinateMatrix[i,0], coordinateMatrix[i,1], coordinateMatrix[i,2], txt)

	# for i in range(len(excl_midpt)):
	# 	txt = 'x'
	# 	ax.text(excl_midpt[i,0], excl_midpt[i,1], excl_midpt[i,2], txt, color='k', fontsize='x-small')

	# ax.set_title('Section of Interest')
	# ax.set_xlabel('X', fontweight ='bold')
	# ax.set_ylabel('Y', fontweight ='bold')
	# ax.set_zlabel('Z', fontweight ='bold')
	# ax.view_init(elev, azim)
	# # cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height]) # vertical bar to the right
	# cax = fig.add_axes([ax.get_position().x0+0.015,ax.get_position().y0-0.05,ax.get_position().width,0.01]) # horiz bar on the bottom
	# plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")
	# plt.show()
	# # exit()

	A = getUnWeightedAdj(N, edges)

	D = np.diag(A.sum(axis=1))

	L = D - A

	# L = compute_mesh_laplacian(V, np.array(F, dtype='int'))

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
