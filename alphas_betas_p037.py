
"""
Requirements: numpy, scipy, scikit-learn
"""
from readMesh import readMesh
from readLAT import readLAT

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

from scipy import spatial
from sklearn.model_selection import KFold
import random
import numpy as np

from utils import *


dataDir = 'data/'
meshNames = ['Patient037_I_MESHData9-RV SINUS VOLTAGE.mesh']
latNames = ['Patient037_I_LATSpatialData_9-RV SINUS VOLTAGE_car.txt']

i = 0

""" Read the files """
meshFile = meshNames[i]
latFile = latNames[i]
nm = meshFile[0:-5]
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[latCoords, latVals] = readLAT(dataDir + latFile)

N = len(coordinateMatrix)	# number of vertices in the graph
M = len(latCoords)			# number of signal samples


""" Map data points to mesh coordinates """
coordKDtree = spatial.cKDTree(coordinateMatrix)
[dist, idxs] = coordKDtree.query(latCoords, k=1)

latVer = coordinateMatrix[idxs]

lat = {}

for i in range(M):
	vertex_num = idxs[i]
	lat_val = latVals[i]
	vertex_coord = coordinateMatrix[vertex_num]
	lat[vertex_num] = {'coord':vertex_coord, 'val':lat_val}


"""
Get edges and corresponding adjacent triangles.

edges: list of edges, edge = set(v_i, v_j)
triangles: list of corr. triangles adj to each edge, tri = (v_i, v_j, v_k)
"""
[edges, triangles] = getE(connectivityMatrix)


""" 
A: NxN unweighted adjacency matrix
W: NxN weighted adjacency matrix (cotan weights)
D: NxN degree matrix
I: NxN identity matrix 

L: NxN Laplace matrix

"""
A = getUnWeightedAdj(coordinateMatrix, edges, triangles)
W = getAdjMatrix(coordinateMatrix, edges, triangles)

D = np.diag(A.sum(axis=1))
I = np.identity(N)

L = D - A

print('Computed A, W, L...')


""" Hyperparameters """
alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
betas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

y = np.zeros((N,1))
M_l = np.zeros((N,N))
M_u = np.zeros((N,N))

folds = 2
kf12 = KFold(n_splits=folds, shuffle=False)

l = 0
best_mse = np.zeros((folds,1))
best_alpha = np.zeros((folds,1))
best_beta = np.zeros((folds,1))
for tr_i, tst_i in kf12.split(idxs):
	print('Fold '+str(l))

	latTrI = np.take(idxs, tr_i)
	latTstI = np.take(idxs, tst_i)

	# plotTrainTestVertices(coordinateMatrix, connectivityMatrix, lat, latTrI, latTstI, nm)

	for i in range(N):
		if i in latTrI:
			y[i] = lat[i]['val']
			M_l[i,i] = float(1)
		else:
			y[i] = 0
			M_u[i,i] = float(1)

	# train
	mse_lst = np.zeros((len(alphas),len(betas)))
	for i in range(len(alphas)):
		alpha = alphas[i]
		for j in range(len(betas)):
			beta = betas[j]
			print('alpha, beta = ' + str(alpha) + ', ' + str(beta))
			
			T = np.linalg.inv(M_l + alpha*M_u + beta*L)

			yhat = np.matmul(T, y)

			mse = 0
			for k in range(N):
				if k in latTstI:
					e = (yhat[k] - lat[k]['val']) ** 2
					mse = mse + e
			mse = mse/len(latTstI)
			mse_lst[i,j] = mse

	
	# with np.printoptions(precision=3, suppress=True):
		# print(mse_lst)
		# print(np.amin(mse_lst))
		# print('alpha = ' + str(alphas[best[0]]) + '\tbeta = ' + str(betas[best[1]]))
		# print('MSE: ' + str(mse_lst[best]))
	
	
	result = np.where(mse_lst == np.amin(mse_lst))
	listOfCordinates = list(zip(result[0], result[1]))
	best = listOfCordinates[0]
	best_mse[l] = mse_lst[best]
	best_alpha[l] = alphas[best[0]]
	best_beta[l] = betas[best[1]]
	l += 1

with np.printoptions(precision=4, suppress=True):
	z = np.array([[a[0], b[0], m[0]] for a, b, m in zip(best_alpha, best_beta, best_mse)])
	print(z)
# result = np.where(best_mse == np.amin(best_mse))
# listOfCordinates = list(zip(result[0], result[1]))
# best = listOfCordinates[0]
# print('alpha = ' + str(best_alpha[best[0]]) + '\tbeta = ' + str(best_beta[best[1]]))
# print('MSE: ' + str(best_mse[best]))



	# plotTrainTestResult(coordinateMatrix, connectivityMatrix, lat, latVer, latVals, latTrI, latTstI, yhat)


	# print(latTrI)
	# print(latTstI)
# print(edges)
# print(A)
# print(W)
# print(D)
# print(I)
# print(L)
# print(U)	# eigenvectors
# print(E)	# eigenvalues
# print(U*E*Ut)
# print(S)
# print(y)
# print(yhat)
	# print(mse)
