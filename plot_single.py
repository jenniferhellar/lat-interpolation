
import numpy as np

# plotting packages
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# KD-Tree for mapping to nearest point
from scipy.spatial import cKDTree

# cross-validation package
from sklearn.model_selection import KFold

# nearest-neighbor interpolation
from scipy.interpolate import griddata

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


from utils import *


WRAP				=		0


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

i = 5

""" Read the files """
meshFile = meshNames[i]
latFile = latNames[i]
nm = meshFile[0:-5]

print('Reading files for ' + nm + ' ...')
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[latCoords, latVals] = readLAT(dataDir + latFile)

N = len(coordinateMatrix)	# number of vertices in the graph
M = len(latCoords)			# number of signal samples

IDX = [i for i in range(N)]
COORD = [coordinateMatrix[i] for i in IDX]

# Map data points to mesh coordinates
coordKDtree = cKDTree(coordinateMatrix)
[dist, idxs] = coordKDtree.query(latCoords, k=1)

IS_SAMP = [False for i in range(N)]
LAT = [0 for i in range(N)]
for i in range(M):
	verIdx = idxs[i]
	IS_SAMP[verIdx] = True
	LAT[verIdx] = latVals[i]

SAMP_IDX = [IDX[i] for i in range(N) if IS_SAMP[i] is True]
SAMP_COORD = [COORD[i] for i in range(N) if IS_SAMP[i] is True]
SAMP_LAT = [LAT[i] for i in range(N) if IS_SAMP[i] is True]

pltCoord = np.array(SAMP_COORD)

# frontI = [i for i in range(len(SAMP_COORD)) if SAMP_COORD[i][1] < 35]
s = np.array(SAMP_COORD)
x = s[:,0]
y = s[:,1]
z = s[:,2]

x_plus_y = x + y
frontI = [i for i in range(len(SAMP_COORD))]
frontCoord = np.array([SAMP_COORD[i] for i in frontI])
frontLAT = [SAMP_LAT[i] for i in frontI]


triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8), subplot_kw=dict(projection="3d"))

ax.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
# ax.plot_trisurf(triang, coordinateMatrix[:,2], color='grey')
pos = ax.scatter(frontCoord[:,0], frontCoord[:,1], frontCoord[:,2], c=frontLAT, cmap='rainbow_r', vmin=-200, vmax=50, s = 20)

ax.set_title('LAT Signal (True)')
ax.set_xlabel('X', fontweight ='bold') 
ax.set_ylabel('Y', fontweight ='bold') 
ax.set_zlabel('Z', fontweight ='bold')
cax = fig.add_axes([ax.get_position().x0+0.015,ax.get_position().y0-0.05,ax.get_position().width,0.01])
plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

plt.show()