from vedo import *
import numpy as np
import os

from readMesh import readMesh
from readLAT import readLAT

from utils import *
from const import *

PATIENT_MAP         =           21

outDir              =           'raw_samples'

meshFile = meshNames[PATIENT_MAP]
latFile = latNames[PATIENT_MAP]
nm = meshFile[0:-5]
patient = nm[7:10]

[vertices, faces] = readMesh(os.path.join(dataDir, meshFile))
[OrigLatCoords, OrigLatVals] = readLAT(os.path.join(dataDir, latFile))

n = len(vertices)
mapIdx = [i for i in range(n)]
mapVerts = [vertices[i] for i in mapIdx]

allLatIdx, allLatVerts, allLatVals = mapSamps(mapIdx, mapVerts, OrigLatCoords, OrigLatVals)

mesh = Mesh([vertices, faces])
# mesh.backColor('white').lineColor('black').lineWidth(0.25)
mesh.c('grey')

origLatPoints = Points(OrigLatCoords, r=10).cmap('rainbow_r', OrigLatVals, vmin=np.min(OrigLatVals), vmax=np.max(OrigLatVals)).addScalarBar()
latPoints = Points(allLatVerts, r=10).cmap('rainbow_r', allLatVals, vmin=np.min(allLatVals), vmax=np.max(allLatVals)).addScalarBar()

# pts2 = mesh.points()[:100]

# scalars = np.random.randint(45, 123, 100)

# points = Points(pts2, r=10).cmap('rainbow', scalars, vmin=45, vmax=123).addScalarBar()

# mesh.interpolateDataFrom(points, N=5).cmap('rainbow').addScalarBar()

show(mesh, origLatPoints, __doc__, axes=9).close()
show(mesh, latPoints, __doc__, axes=9).close()

# show(mesh, __doc__, axes=9).close()
