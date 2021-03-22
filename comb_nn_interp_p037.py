
"""
Requirements: numpy, scipy, scikit-learn
"""
from readMesh import readMesh
from readLAT import readLAT

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

from scipy import spatial
from scipy.interpolate import griddata
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

print('Reading files for ' + nm + ' ...')
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
print('Generating edge matrix ...')
[edges, triangles] = getE(connectivityMatrix)


""" 
A: NxN unweighted adjacency matrix
W: NxN weighted adjacency matrix (cotan weights)
D: NxN degree matrix
I: NxN identity matrix 

L: NxN Laplace matrix

"""
print('Calculating adjacency matrices ...')
A = getUnWeightedAdj(coordinateMatrix, edges, triangles)
W = getAdjMatrix(coordinateMatrix, edges, triangles)

D = np.diag(A.sum(axis=1))
I = np.identity(N)

print('Calculating Laplacian matrix ...')
# L = D - W
L = D - A


""" Hyperparameters """
alpha = 0.1
beta = 1

y = np.zeros((N,1))
M_l = np.zeros((N,N))
M_u = np.zeros((N,N))

folds = 10
kf12 = KFold(n_splits=folds, shuffle=True)

l = 0
errorVecNN = np.zeros((N,1))
errorVecOur = np.zeros((N,1))
y_out_our = np.zeros((N,1))
for tr_i, tst_i in kf12.split(idxs):
	print('\nFold ' + str(l) + '\t# of labelled vertices: ' + str(len(tr_i)) + '\t# of unlabelled vertices: ' + str(len(tst_i)))

	latTrI = np.take(idxs, tr_i)
	latTstI = np.take(idxs, tst_i)

	TestValList = []
	for i in latTstI:
		TestValList.append(lat[i]['val'])

	TrainVerCoord = []
	TrainValList = []
	for i in latTrI:
		TrainVerCoord.append(lat[i]['coord'])
		TrainValList.append(lat[i]['val'])

	TestVerCoord = []
	for i in latTstI:
		TestVerCoord.append(lat[i]['coord'])

	TrainCoordList = np.array(TrainVerCoord)
	TestCoordList = np.array(TestVerCoord)

	TrainValList = np.array(TrainValList)

	nn_out = griddata(TrainCoordList, TrainValList, TestCoordList, method='nearest')
	nnmse = 0
	nn_err = []
	for k in range(len(nn_out)):
		err = nn_out[k] - TestValList[k]
		nn_err.append(err)
		e = err **2
		nnmse = nnmse + e

		errorVecNN[latTstI[k]] = err
	nnmse = math.sqrt(nnmse/len(nn_err))
	print('NN-Estimate RMSE:\t' + str(nnmse))


	for i in range(N):
		if i in latTrI:
			y[i] = lat[i]['val']
			M_l[i,i] = float(1)
		else:
			y[i] = 0
			M_u[i,i] = float(1)

			
	T = np.linalg.inv(M_l + alpha*M_u + beta*L)

	yhat = np.matmul(T, y)

	out = yhat[latTstI]

	mse = 0
	our_err = []
	for k in range(len(out)):
		err = out[k] - TestValList[k]
		our_err.append(err)
		e = err **2
		mse = mse + e

		errorVecOur[latTstI[k]] = err
		y_out_our[latTstI[k]] = out[k]
	mse = math.sqrt(mse/len(our_err))
	print('Our Estimate RMSE:\t' + str(mse))

	# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
	# axes = ax.flatten()

	# x = [i for i in range(len(nn_err))]
	# nn_err2 = [i**2 for i in nn_err]
	# our_err2 = [i**2 for i in our_err]

	# thisAx = axes[0]
	# thisAx.scatter(x, nn_err)
	# thisAx.set_title('NN Error')
	# thisAx = axes[1]
	# thisAx.scatter(x, nn_err2)
	# thisAx.set_title('(NN Error)^2')
	# thisAx = axes[2]

	# x = [i for i in range(len(our_err))]
	# thisAx.scatter(x, our_err)
	# thisAx.set_title('Our Error')
	# thisAx = axes[3]
	# thisAx.scatter(x, our_err2)
	# thisAx.set_title('(Our Error)^2')
	# plt.show()

	# nm = meshFile[0:-5]+'\nFold '+str(l)+'\nNN-MSE = '+str(nnmse)+', Our-MSE = '+str(mse[0])
	# plotTrainTestVertices(coordinateMatrix, connectivityMatrix, lat, latTrI, latTstI, nm)


	l += 1

	# plotTrainTestResult(coordinateMatrix, connectivityMatrix, lat, latVer, latVals, latTrI, latTstI, yhat)

errorVecNN = errorVecNN[idxs]
errorVecOur = errorVecOur[idxs]

y_out_our = y_out_our[idxs]

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize=(16,8))
axes = ax.flatten()

x = [i for i in range(M)]
thisAx = axes[0]
thisAx.scatter(x, errorVecNN)
thisAx.set_title('NN Error')
thisAx = axes[1]
thisAx.scatter(x, errorVecOur)
thisAx.set_title('Our Error')
plt.show()


n, bins, patches = plt.hist(x=errorVecOur, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Error (Our Method)')
# plt.text(23, 45, r'$\alpha=0.1, \beta=1$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()


fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()

thisAx = axes[0]
triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)
thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(latVer[:,0], latVer[:,1], latVer[:,2], c=latVals, cmap='rainbow_r', s = 20)
thisAx.set_title('LAT Signal (True)')

cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height])
plt.colorbar(pos, cax=cax) # Similar to fig.colorbar(im, cax = cax)

thisAx = axes[1]
triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)
thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(latVer[:,0], latVer[:,1], latVer[:,2], c=y_out_our, cmap='rainbow_r', s = 20)
thisAx.set_title('LAT Signal (Estimated)')

cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height])
plt.colorbar(pos, cax=cax) # Similar to fig.colorbar(im, cax = cax)

thisAx = axes[2]
thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(latVer[:,0], latVer[:,1], latVer[:,2], c=errorVecOur, cmap='rainbow_r', s = 20)
thisAx.set_title('Estimation Error')

cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height])
plt.colorbar(pos, cax=cax) # Similar to fig.colorbar(im, cax = cax)

plt.show()