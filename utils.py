
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


def getE(connectivityMatrix):
	edges = []
	triangles = []
	for tri in connectivityMatrix:

		idx0 = int(tri[0])
		idx1 = int(tri[1])
		idx2 = int(tri[2])

		e1 = set([idx0, idx1])
		e2 = set([idx1, idx2])
		e3 = set([idx0, idx2])

		if e1 not in edges:
			edges.append(e1)
			triangles.append([tri])
		else:
			k = edges.index(e1)
			triangles[k].append(tri)

		if e2 not in edges:
			edges.append(e2)
			triangles.append([tri])
		else:
			k = edges.index(e2)
			triangles[k].append(tri)

		if e3 not in edges:
			edges.append(e3)
			triangles.append([tri])
		else:
			k = edges.index(e3)
			triangles[k].append(tri)

	# for tri2 in triangles:
	# 	if len(tri2) != 2:
	# 		print(tri2)

	return [np.array(edges), triangles]


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