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


def getE(coordinateMatrix, connectivityMatrix, LAT, thresh):
	""" 
	Computes a list of edges in the graph, based on the triangles in
	connectivityMatrix, removing edges with a deltaLAT > thresh.  Returns
	a numpy array of the edges and a list of midpoints of removed edges,
	computed based on vertex coordinates in cooordinateMatrix.
	"""
	edges = []

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
				# if removed, compute midpoint
				else:
					[x1, y1, z1] = coordinateMatrix[idx0, :]
					[x2, y2, z2] = coordinateMatrix[idx1, :]
					[x, y, z] = [float(x1+x2)/2, float(y1+y2)/2, float(z1+z2)/2]
					excl_edges.append(e1)
					excl_midpt.append([x, y, z])

		# repeat for second edge
		if (e2 not in edges):
			if (e2 not in excl_edges):
				if (e2_lat < thresh):
					edges.append(e2)
				else:
					[x1, y1, z1] = coordinateMatrix[idx1, :]
					[x2, y2, z2] = coordinateMatrix[idx2, :]
					[x, y, z] = [float(x1+x2)/2, float(y1+y2)/2, float(z1+z2)/2]
					excl_edges.append(e2)
					excl_midpt.append([x, y, z])

		# repeat for third edge
		if (e3 not in edges):
			if (e3 not in excl_edges):
				if (e3_lat < thresh):
					edges.append(e3)
				else:
					[x1, y1, z1] = coordinateMatrix[idx0, :]
					[x2, y2, z2] = coordinateMatrix[idx2, :]
					[x, y, z] = [float(x1+x2)/2, float(y1+y2)/2, float(z1+z2)/2]
					excl_edges.append(e3)
					excl_midpt.append([x, y, z])

	return [np.array(edges), np.array(excl_midpt)]


def getUnWeightedAdj(N, edges):
	""" Computes the binary adjacency matrix """
	A = np.zeros((N, N))

	for i in range(len(edges)):

		e = list(edges[i])	# list of edges
		v_i = e[0]	# vertices given as indices
		v_j = e[1]

		A[v_i, v_j] = 1
		A[v_j, v_i] = 1

	return A