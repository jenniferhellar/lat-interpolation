
from readMesh import readMesh
from readLAT import readLAT

import math

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

from scipy import spatial

dataDir = 'data/'

meshNames = ['Patient031_I_MESHData4-SINUS LVFAM.mesh', 
				'Patient032_I_MESHData4-LVFAM SINUS.mesh',
				'Patient033_I_MESHData4-RV FAM PVC A - LAT HYBRID.mesh', 
				'Patient034_I_MESHData6-RVFAM SINUS VOLTAGE.mesh', 
				'Patient035_I_MESHData8-SINUS.mesh',
				'Patient037_I_MESHData9-RV SINUS VOLTAGE.mesh']

latNames = ['Patient031_I_LATSpatialData_4-SINUS LVFAM_car.txt',
			'Patient032_I_LATSpatialData_4-LVFAM SINUS_car.txt',
			'Patient033_I_LATSpatialData_4-RV FAM PVC A - LAT HYBRID_car.txt',
			'Patient034_I_LATSpatialData_6-RVFAM SINUS VOLTAGE_car.txt',
			'Patient035_I_LATSpatialData_8-SINUS_car.txt',
			'Patient037_I_LATSpatialData_9-RV SINUS VOLTAGE_car.txt']

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()

i = 5

meshFile = meshNames[i]
latFile = latNames[i]
nm = meshFile[0:-5]
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[latCoords, latVals] = readLAT(dataDir + latFile)

coordKDtree = spatial.cKDTree(coordinateMatrix)
[dist, idxs] = coordKDtree.query(latCoords, k=1)

latVer = coordinateMatrix[idxs]

minLat = math.floor(min(latVals)/10)*10
maxLat = math.ceil(max(latVals)/10)*10

# print(len(latCoords), len(coordinateMatrix), len(connectivityMatrix))
# print(max(latVals), min(latVals))

triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

thisAx = axes[0]
thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
thisAx.scatter(latCoords[:,0], latCoords[:,1], latCoords[:,2], c=latVals, cmap='rainbow_r', vmin=minLat, vmax=maxLat, s = 10)
thisAx.set_title('Original LAT Coordinates')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')

thisAx = axes[1]
thisAx.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
pos = thisAx.scatter(latVer[:,0], latVer[:,1], latVer[:,2], c=latVals, cmap='rainbow_r', vmin=minLat, vmax=maxLat, s = 10)
thisAx.set_title('LAT Value Assigned to Nearest Mesh Vertex')
thisAx.set_xlabel('X', fontweight ='bold') 
thisAx.set_ylabel('Y', fontweight ='bold') 
thisAx.set_zlabel('Z', fontweight ='bold')

cax = fig.add_axes([thisAx.get_position().x1+0.03,thisAx.get_position().y0,0.01,thisAx.get_position().height])
plt.colorbar(pos, cax=cax, label='LAT (ms)') # Similar to fig.colorbar(im, cax = cax)

plt.show()