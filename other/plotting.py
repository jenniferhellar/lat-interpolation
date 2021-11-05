"""
DATA INDICES:
	Too large for my laptop:
		p031 = 0 (4-SINUS LVFAM)
		p032 = 1 (1-LVFAM LAT HYB), 2 (2-LVFAM INITIAL PVC), 3 (4-LVFAM SINUS)
		p037 = 10 (12-LV-SINUS)

	Testable:
		p033 = 4 (3-RV-FAM-PVC-A-NORMAL), 5 (4-RV-FAM-PVC-A-LAT-HYBRID)
		p034 = 6 (4-RVFAM-LAT-HYBRID), 7 (5-RVFAM-PVC), 8 (6-RVFAM-SINUS-VOLTAGE)
		p035 = 9 (8-SINUS)
		p037 = 11 (9-RV-SINUS-VOLTAGE)
"""


from vedo import *
import numpy as np
import os
import argparse

from readMesh import readMesh
from readLAT import readLAT

from utils import *
from const import DATADIR, DATAFILES

OUTDIR              =           'plotting_results'

""" Parse the input for data index argument. """
parser = argparse.ArgumentParser(
    description='Processes a single mesh file repeatedly for comparison of MAGIC-LAT, GPR, and quLATi performance.')

parser.add_argument('-i', '--idx', required=True, default='11',
                    help='Data index to process. \
                    Default: 11')

args = parser.parse_args()

PATIENT_IDX				=		int(vars(args)['idx'])

""" Obtain file names, patient number, mesh id, etc. """
(meshFile, latFile, ablFile) = DATAFILES[PATIENT_IDX]
nm = meshFile[0:-5]
patient = nm[7:10]
id = latFile.split('_')[3]

""" Create output directory for this script and subdir for this mesh. """
outSubDir = os.path.join(OUTDIR, 'p' + patient + '_' + id)
if not os.path.isdir(OUTDIR):
	os.makedirs(OUTDIR)
if not os.path.isdir(outSubDir):
	os.makedirs(outSubDir)

""" Read the files """
print('\nProcessing ' + nm + ' ...\n')
[vertices, faces] = readMesh(os.path.join(DATADIR, meshFile))
[OrigLatCoords, OrigLatVals] = readLAT(os.path.join(DATADIR, latFile))

if ablFile != '':
	ablFile = os.path.join(DATADIR, ablFile)
else:
	ablFile = None
	print('No ablation file available for this mesh... continuing...\n')

n = len(vertices)
mapIdx = [i for i in range(n)]
mapVerts = [vertices[i] for i in mapIdx]

allLatIdx, allLatVerts, allLatVals = mapSamps(mapIdx, mapVerts, OrigLatCoords, OrigLatVals)
# For colorbar ranges
MINLAT = math.floor(min(allLatVals)/10)*10
# MAXLAT = math.ceil(max(latVals)/10)*10
MAXLAT = MINLAT + math.ceil((7/8 * (max(allLatVals) - MINLAT)) / 10)*10

mesh = Mesh([vertices, faces])
# mesh.backColor('white').lineColor('black').lineWidth(0.25)
mesh.c('grey')
r = 10
size = (100, 800)
fontSize = 35

origLatPoints = Points(OrigLatCoords, r=r).cmap('gist_rainbow', OrigLatVals, vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white', title='LAT (ms)   ', titleFontSize=fontSize, size=size)
latPoints = Points(allLatVerts, r=r).cmap('gist_rainbow', allLatVals, vmin=MINLAT, vmax=MAXLAT).addScalarBar(c='white', title='LAT (ms)   ', titleFontSize=fontSize, size=size)

# pts2 = mesh.points()[:100]

# scalars = np.random.randint(45, 123, 100)

# points = Points(pts2, r=10).cmap('rainbow', scalars, vmin=45, vmax=123).addScalarBar()

# mesh.interpolateDataFrom(points, N=5).cmap('rainbow').addScalarBar()

# show(mesh, origLatPoints, __doc__, axes=9).close()
# show(mesh, origLatPoints, bg='black', azimuth=160, elevation=0, roll=0).close()
show(mesh, latPoints, bg='black', azimuth=160, elevation=0, roll=0).close()
exit(0)


""" Testing various perspectives """

vplt = Plotter(N=1, axes=0, interactive=True)
elev = 0
roll = 0
azim = [0, 90, 180, 270]
for a in azim:
	vplt.show(mesh, latPoints, azimuth=a, elevation=elev, roll=roll, bg='black')
	vplt.screenshot(filename=os.path.join(outSubDir, 'elev{:g}azim{:g}'.format(elev, a)), returnNumpy=False)
elev = [-90, 90]
roll = 0
azim = 0
for e in elev:
	vplt.show(mesh, latPoints, azimuth=azim, elevation=e, roll=roll, bg='black')
	vplt.screenshot(filename=os.path.join(outSubDir, 'elev{:g}azim{:g}'.format(e, azim)), returnNumpy=False)
vplt.close()

# elev = [-90, 90]
# roll = 0
# azim = 0
# for e in elev:
# 	vplt = Plotter(N=1, axes=0, offscreen=True)
# 	vplt.show(mesh, latPoints, azimuth=azim, elevation=e, roll=roll, bg='black')
# 	vplt.screenshot(filename=os.path.join(outSubDir, 'elev{:g}azim{:g}'.format(e, azim)), returnNumpy=False)
# 	vplt.close()


""" Old """

# # plotting packages
# import matplotlib.pyplot as plt
# import matplotlib.tri as mtri

# triang = mtri.Triangulation(vertices[:,0], vertices[:,1], triangles=faces)

# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16, 8), subplot_kw=dict(projection="3d"))

# ax.plot_trisurf(triang, vertices[:,2], color='grey', alpha=0.2)
# # ax.plot_trisurf(triang, coordinateMatrix[:,2], color='grey')
# allLatVerts = np.array(allLatVerts)
# pos = ax.scatter(allLatVerts[:,0], allLatVerts[:,1], allLatVerts[:,2], c=allLatVals, cmap='rainbow_r', vmin=np.min(allLatVals), vmax=np.max(allLatVals), s = 20)

# ax.set_title('LAT Signal (True)')
# ax.set_xlabel('X', fontweight ='bold') 
# ax.set_ylabel('Y', fontweight ='bold') 
# ax.set_zlabel('Z', fontweight ='bold')
# cax = fig.add_axes([ax.get_position().x0+0.015,ax.get_position().y0-0.05,ax.get_position().width,0.01])
# plt.colorbar(pos, cax=cax, label='LAT (ms)', orientation="horizontal")

# plt.show()
