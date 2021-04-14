
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import norm
import math

def plotTrainTestVertices(coordinateMatrix, connectivityMatrix, lat, latTrI, latTstI, nm):
	TrainVerCoord = {}
	TrainVerLAT = {}
	for i in latTrI:
		TrainVerCoord[i] = lat[i]['coord']
		TrainVerLAT[i] = lat[i]['val']

	TestVerCoord = {}
	TestVerLAT = {}
	for i in latTstI:
		TestVerCoord[i] = lat[i]['coord']
		TestVerLAT[i] = lat[i]['val']

	TrainCoordList = np.array(list(TrainVerCoord.values()))
	TestCoordList = np.array(list(TestVerCoord.values()))

	fig, ax = plt.subplots(figsize=(16, 8), subplot_kw=dict(projection="3d"))

	triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)
	ax.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	ax.scatter(TrainCoordList[:,0], TrainCoordList[:,1], TrainCoordList[:,2], c='blue', s = 20)
	ax.scatter(TestCoordList[:,0], TestCoordList[:,1], TestCoordList[:,2], c='red', s = 20)
	ax.set_title(nm)

	plt.show()


def plotTrainTestResult(coordinateMatrix, connectivityMatrix, lat, latVer, latVals, latTrI, latTstI, yhat):
	TrainVerCoord = {}
	TrainVerLAT = {}
	for i in latTrI:
		TrainVerCoord[i] = lat[i]['coord']
		TrainVerLAT[i] = lat[i]['val']

	TestVerCoord = {}
	TestVerLAT = {}
	for i in latTstI:
		TestVerCoord[i] = lat[i]['coord']
		TestVerLAT[i] = yhat[i]

	TrainCoordList = np.array(list(TrainVerCoord.values()))
	TestCoordList = np.array(list(TestVerCoord.values()))

	TrainValList = np.array(list(TrainVerLAT.values()))
	TestValList = np.array(list(TestVerLAT.values()))

	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 8), subplot_kw=dict(projection="3d"))
	axes = ax.flatten()

	triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

	thisAx = axes[0]
	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(latVer[:,0], latVer[:,1], latVer[:,2], c=latVals, cmap='rainbow_r', s = 10)
	thisAx.set_title('Ground truth')

	thisAx = axes[1]
	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(TrainCoordList[:,0], TrainCoordList[:,1], TrainCoordList[:,2], c=TrainValList, cmap='rainbow_r', s = 10)
	thisAx.set_title('Given (y)')

	thisAx = axes[2]
	thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
	pos = thisAx.scatter(TestCoordList[:,0], TestCoordList[:,1], TestCoordList[:,2], c=TestValList, cmap='rainbow_r', s = 10)
	thisAx.set_title('Test Output')
	cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height])
	plt.colorbar(pos, cax=cax) # Similar to fig.colorbar(im, cax = cax)

	plt.show()


def pltEigenvalues(lambdas):
	N = len(lambdas)
	plt.scatter(range(0, N), lambdas)
	plt.xlabel('Eigenvalue Index')
	plt.ylabel('Eigenvalue')
	plt.title('Graph Laplacian Eigenvalues')
	plt.show()


def getE(coordinateMatrix, connectivityMatrix, LAT, thresh):
	edges = []
	triangles = []

	excl_edges = []
	excl_midpt = []
	for tri in connectivityMatrix:

		idx0 = int(tri[0])
		idx1 = int(tri[1])
		idx2 = int(tri[2])

		e1 = set([idx0, idx1])
		e1_lat = abs(LAT[idx0] - LAT[idx1])
		e2 = set([idx1, idx2])
		e2_lat = abs(LAT[idx1] - LAT[idx2])
		e3 = set([idx0, idx2])
		e3_lat = abs(LAT[idx0] - LAT[idx2])

		if (e1 not in edges):
			if (e1 not in excl_edges):
				if (e1_lat < thresh):
					edges.append(e1)
					triangles.append([tri])
				else:
					[x1, y1, z1] = coordinateMatrix[idx0, :]
					[x2, y2, z2] = coordinateMatrix[idx1, :]
					[x, y, z] = [float(x1+x2)/2, float(y1+y2)/2, float(z1+z2)/2]
					excl_edges.append(e1)
					excl_midpt.append([x, y, z])
		else:
			k = edges.index(e1)
			triangles[k].append(tri)

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


def getUnWeightedAdj(coordinateMatrix, edges, triangles):
	N = len(coordinateMatrix)
	A = np.zeros((N, N))

	for i in range(len(edges)):

		e = list(edges[i])
		v_i = e[0]
		v_j = e[1]

		A[v_i, v_j] = 1
		A[v_j, v_i] = 1

	return A


def getAdjMatrix(coordinateMatrix, edges, triangles):
	N = len(coordinateMatrix)
	A = np.zeros((N, N))

	for i in range(len(edges)):

		e = edges[i]
		
		e = list(e)	
		v_i = e[0]
		v_j = e[1]

		[x1, y1, z1] = coordinateMatrix[v_i, :]
		[x2, y2, z2] = coordinateMatrix[v_j, :]

		w_ij = 1/math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

		A[v_i, v_j] = w_ij
		A[v_j, v_i] = w_ij

	return A


def getAdjMatrixCotan(coordinateMatrix, edges, triangles):
	N = len(coordinateMatrix)
	A = np.zeros((N, N))

	for i in range(len(edges)):

		e = edges[i]
		adj_tri = triangles[i]

		w_ij = 0

		for tri in adj_tri:
			idx0 = int(tri[0])
			idx1 = int(tri[1])
			idx2 = int(tri[2])

			pt0 = coordinateMatrix[idx0, :]
			pt1 = coordinateMatrix[idx1, :]
			pt2 = coordinateMatrix[idx2, :]

			l_a = norm(pt1 - pt0)
			l_b = norm(pt2 - pt0)
			l_c = norm(pt2 - pt1)

			s = (l_a + l_b + l_c)/2

			a = math.sqrt(s*(s-l_a)*(s-l_b)*(s-l_c))

			if set([idx0, idx1]) == e:
				w_ij = w_ij + (-l_a**2 + l_b**2 + l_c**2)/(8*a)
			elif set([idx1, idx2]) == e:
				w_ij = w_ij + (l_a**2 + l_b**2 - l_c**2)/(8*a)
			elif set([idx0, idx2]) == e:
				w_ij = w_ij + (l_a**2 - l_b**2 + l_c**2)/(8*a)
			else:
				print('unable to identify edge')
				exit()
		
		e = list(e)	
		v_i = e[0]
		v_j = e[1]

		A[v_i, v_j] = w_ij
		A[v_j, v_i] = w_ij

	return A


def getAdjMatrixExp(coordinateMatrix, edges, triangles):
	N = len(coordinateMatrix)
	A = np.zeros((N, N))

	d = 0
	cnt = 0
	for i in range(len(coordinateMatrix)):
		[x1, y1, z1] = coordinateMatrix[i, :]
		for j in range(len(coordinateMatrix)):
			[x2, y2, z2] = coordinateMatrix[j, :]
			d += math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
			cnt += 1
	d = d/cnt

	for i in range(len(edges)):

		e = edges[i]
		
		e = list(e)	
		v_i = e[0]
		v_j = e[1]

		[x1, y1, z1] = coordinateMatrix[v_i, :]
		[x2, y2, z2] = coordinateMatrix[v_j, :]

		# w_ij = 1/math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
		d2_ij = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
		w_ij = math.exp(-d2_ij/d**2)

		A[v_i, v_j] = w_ij
		A[v_j, v_i] = w_ij

	return A


def pltAdjMatrix(A, first, numV, title):

	end = first + numV + 1;

	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8))
	pos = ax.imshow(A[first:end, first:end], cmap='Blues', interpolation=None)
	plt.xticks(np.arange(first, end, step=1))
	plt.yticks(np.arange(first, end, step=1))
	plt.title(title + '\nVertices ' + str(first) + ' - ' + str(end-1))
	# plt.colorbar(hm)
	# cax = fig.add_axes([ax.get_position().x0,ax.get_position().y0-0.1,ax.get_position().width,0.01])
	cax = fig.add_axes([ax.get_position().x1+0.03,ax.get_position().y0,0.01,ax.get_position().height])
	plt.colorbar(pos, cax=cax, label='weight')
	# plt.show()