
"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""

import os

import numpy as np
import math
import random

# plotting packages
from vedo import *

# loading images
import cv2

# Gaussian process regression interpolation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


from utils import *
from const import *
from magicLAT import *



NUM_TRAIN_SAMPS 		= 		200
EDGE_THRESHOLD			=		50

outDir				 	=		'results_wip'

mesh = Sphere(pos=(0,0,0), r=1, c='grey', quads=False).lw(0.2)

vertices = mesh.points()
faces = mesh.faces()
[edges, _] = edgeMatrix(vertices, faces)

x, y, z = np.split(vertices, 3, axis=1)
lat = np.zeros(len(x))
for i in range(len(lat)):
	if z[i] > 0 and y[i] > 0 and x[i] > 0:
		lat[i] = sin(x[i]) - sin(y[i])
	else:
		lat[i] = sin(x[i]+np.pi/2) + sin(y[i])
lat = np.round(lat*100)

verPoints = Points(vertices, r=10).cmap('rainbow_r', lat).addScalarBar()
# mesh.cmap('rainbow_r', lat).addScalarBar()
show(mesh, verPoints, axes=4).close()

n = len(vertices)

mapIdx = [i for i in range(n)]
mapCoord = [vertices[i] for i in mapIdx]
mapLAT = [lat[i] for i in mapIdx]

if not os.path.isdir(outDir):
	os.makedirs(outDir)

print('{:<20}{:g}'.format('n', n))
print('{:<20}{:g}/{:g}'.format('m', NUM_TRAIN_SAMPS, n))


""" Do the interpolation """

# get vertex indices of labelled/unlabelled nodes
TrIdx = random.sample(mapIdx, NUM_TRAIN_SAMPS)
TstIdx = [i for i in mapIdx if i not in TrIdx]

# get vertex coordinates
TrCoord = [vertices[i] for i in TrIdx]
TstCoord = [vertices[i] for i in TstIdx]

# get mapLAT signal values
TrVal = [lat[i] for i in TrIdx]
TstVal = [lat[i] for i in TstIdx]


""" MAGIC-LAT estimate """
latEst = magicLAT(vertices, faces, edges, TrIdx, TrCoord, TrVal, EDGE_THRESHOLD)

""" Create GPR kernel and regressor """
gp_kernel = RBF(length_scale=0.01) + RBF(length_scale=0.1) + RBF(length_scale=1)
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)

""" GPR estimate """
# fit the GPR with training samples
gpr.fit(TrCoord, TrVal)

# predict the entire signal
latEstGPR = gpr.predict(vertices, return_std=False)


# For colorbar ranges
MINLAT = math.floor(min(lat)/10)*10
MAXLAT = math.ceil(max(lat)/10)*10

elev = 0
azimuth = 0
roll = 0

"""
Figure 0: Ground truth (entire), training points, and MAGIC-LAT (entire)
"""
plt = Plotter(N=3, axes=9)

# Plot 0: Ground truth
plt.show(mesh, verPoints, 'all known points', azimuth=azimuth, elevation=elev, roll=roll, at=0)

# Plot 1: Training points
trainPoints = Points(TrCoord, r=10).cmap('rainbow_r', TrVal, vmin=MINLAT, vmax=MAXLAT).addScalarBar()
plt.show(mesh, trainPoints, 'training points', at=1)

# Plot 2: MAGIC-LAT output signal
magicPoints = Points(vertices, r=10).cmap('rainbow_r', latEst, vmin=MINLAT, vmax=MAXLAT).addScalarBar()

plt.show(mesh, magicPoints, 'interpolation result', title='MAGIC-LAT', at=2, interactive=True)
plt.screenshot(filename=os.path.join(outDir, 'magic.png'), returnNumpy=False)
plt.close()


"""
Figure 1: Ground truth (entire), training points, and GPR (entire)
"""
plt = Plotter(N=3, axes=9)

# Plot 0: Ground truth
plt.show(mesh, verPoints, 'all known points', azimuth=azimuth, elevation=elev, roll=roll, at=0)
# Plot 1: Training points
plt.show(mesh, trainPoints, 'training points', at=1)
# Plot 2: GPR output signal
gprPoints = Points(vertices, r=10).cmap('rainbow_r', latEstGPR, vmin=MINLAT, vmax=MAXLAT).addScalarBar()

plt.show(mesh, gprPoints, 'interpolation result', title='GPR', at=2, interactive=True)
plt.screenshot(filename=os.path.join(outDir, 'gpr.png'), returnNumpy=False)
plt.close()

# mesh.interpolateDataFrom(pts, N=1).cmap('rainbow_r').addScalarBar()

elev = 0
azim = [0, 90, 180, 270]
roll = 0

"""
Figure 2: Ground truth (test points only - for ssim)
"""
testPoints = Points(TstCoord, r=10).cmap('rainbow_r', TstVal, vmin=MINLAT, vmax=MAXLAT)
plt = Plotter(N=1, axes=0)
for a in azim:
	plt.show(mesh, testPoints, azimuth=a, elevation=elev, roll=roll, title='true, azimuth={:g}'.format(a))
	plt.screenshot(filename=os.path.join(outDir, 'true{:g}.png'.format(a)), returnNumpy=False)

"""
Figure 3: MAGIC-LAT estimate (test points only - for ssim)
"""
testEst = Points(TstCoord, r=10).cmap('rainbow_r', latEst[TstIdx], vmin=MINLAT, vmax=MAXLAT)

for a in azim:
	plt.show(mesh, testEst, azimuth=a, elevation=elev, roll=roll, title='MAGIC-LAT, azimuth={:g}'.format(a))
	plt.screenshot(filename=os.path.join(outDir, 'estimate{:g}.png'.format(a)), returnNumpy=False)

"""
Figure 4: GPR estimate (test points only - for ssim)
"""
testEstGPR = Points(TstCoord, r=10).cmap('rainbow_r', latEstGPR[TstIdx], vmin=MINLAT, vmax=MAXLAT)

for a in azim:
	plt.show(mesh, testEstGPR, azimuth=a, elevation=elev, roll=roll, title='GPR, azimuth={:g}'.format(a))
	plt.screenshot(filename=os.path.join(outDir, 'estimateGPR{:g}.png'.format(a)), returnNumpy=False)

plt.close()


"""
Figure 5: quLATi estimate (test points only for ssim)
"""
# TODO


"""
Error metrics
"""
nmse = calcNMSE(TstVal, latEst[TstIdx])
nmseGPR = calcNMSE(TstVal, latEstGPR[TstIdx])

mae = calcMAE(TstVal, latEst[TstIdx])
maeGPR = calcMAE(TstVal, latEstGPR[TstIdx])

ssim = 0
ssimGPR = 0
for a in azim:
	figTruth = cv2.imread(os.path.join(outDir, 'true{:g}.png'.format(a)))
	figEst = cv2.imread(os.path.join(outDir, 'estimate{:g}.png'.format(a)))
	figEstGPR = cv2.imread(os.path.join(outDir, 'estimateGPR{:g}.png'.format(a)))

	ssim += calcSSIM(figTruth, figEst)
	ssimGPR += calcSSIM(figTruth, figEstGPR)

ssim = ssim / len(azim)
ssimGPR = ssimGPR / len(azim)

with open(os.path.join(outDir, 'metrics_ex.txt'), 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}\n\n'.format('Metric', 'MAGIC-LAT', 'GPR'))
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('NMSE', nmse, nmseGPR))
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('MAE', mae, maeGPR))
	fid.write('{:<20}{:<20.6f}{:<20.6f}'.format('SSIM', ssim, ssimGPR))
