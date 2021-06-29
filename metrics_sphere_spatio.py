
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
	if z[i] > 0:
		lat[i] = sin(x[i]) - sin(y[i])
	else:
		lat[i] = sin(y[i]) - sin(z[i])
lat = np.round(lat*100)

verPoints = Points(vertices, r=10).cmap('rainbow_r', lat).addScalarBar()
# mesh.cmap('rainbow_r', lat).addScalarBar()
# show(mesh, verPoints, axes=4).close()

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
plt = Plotter(N=3, axes=9, offscreen=True)

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
plt = Plotter(N=3, axes=9, offscreen=True)

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
whitemesh = Mesh([vertices, faces], c='black')
"""
Figure 2: Ground truth (test points only - for ssim)
"""
plt = Plotter(N=1, axes=0, offscreen=True)
testPoints = Points(TstCoord, r=20).cmap('rainbow_r', TstVal, vmin=MINLAT, vmax=MAXLAT)
for a in azim:
	plt.show(whitemesh, testPoints, azimuth=a, elevation=elev, roll=roll, title='true, azimuth={:g}'.format(a), bg='black')
	plt.screenshot(filename=os.path.join(outDir, 'true{:g}.png'.format(a)), returnNumpy=False)

plt.close()

"""
Figure 3: MAGIC-LAT estimate (test points only - for ssim)
"""
plt = Plotter(N=1, axes=0, offscreen=True)
testEst = Points(TstCoord, r=20).cmap('rainbow_r', latEst[TstIdx], vmin=MINLAT, vmax=MAXLAT)

for a in azim:
	plt.show(whitemesh, testEst, azimuth=a, elevation=elev, roll=roll, title='MAGIC-LAT, azimuth={:g}'.format(a), bg='black')
	plt.screenshot(filename=os.path.join(outDir, 'estimate{:g}.png'.format(a)), returnNumpy=False)

plt.close()

"""
Figure 4: GPR estimate (test points only - for ssim)
"""
plt = Plotter(N=1, axes=0, offscreen=True)
testEstGPR = Points(TstCoord, r=20).cmap('rainbow_r', latEstGPR[TstIdx], vmin=MINLAT, vmax=MAXLAT)

for a in azim:
	plt.show(whitemesh, testEstGPR, azimuth=a, elevation=elev, roll=roll, title='GPR, azimuth={:g}'.format(a), bg='black')
	plt.screenshot(filename=os.path.join(outDir, 'estimateGPR{:g}.png'.format(a)), returnNumpy=False)

plt.close()


"""
Figure 5: quLATi estimate (test points only for ssim)
"""
# TODO


"""
Error metrics
"""
bin_edges = np.linspace(start=MINLAT, stop=MAXLAT, num=21, endpoint=True)

nTst, bins = np.histogram(TstVal, bins=bin_edges)

n, bins = np.histogram(latEst[TstIdx], bins=bin_edges)

nGPR, bins = np.histogram(latEstGPR[TstIdx], bins=bin_edges)

print(calcNMSE(nTst, n))
print(calcNMSE(nTst, nGPR))

nmse = calcNMSE(TstVal, latEst[TstIdx])
nmseGPR = calcNMSE(TstVal, latEstGPR[TstIdx])

mae = calcMAE(TstVal, latEst[TstIdx])
maeGPR = calcMAE(TstVal, latEstGPR[TstIdx])

from matplotlib import pyplot as plt
b = [0, 0]
g = [0, 0]
r = [0, 0]

bins = 256

for a in azim:
	img = cv2.imread(os.path.join(outDir, 'true{:g}.png'.format(a)), cv2.IMREAD_GRAYSCALE)
	n_black_px = np.sum(img == 0)
	N = np.sum(img > 0)

	figTruth = cv2.imread(os.path.join(outDir, 'true{:g}.png'.format(a)))
	figEst = cv2.imread(os.path.join(outDir, 'estimate{:g}.png'.format(a)))
	figEstGPR = cv2.imread(os.path.join(outDir, 'estimateGPR{:g}.png'.format(a)))

	# Calculate histograms
	hist1 = cv2.calcHist([figTruth],[0],None,[bins],[0,256])
	hist2 = cv2.calcHist([figTruth],[1],None,[bins],[0,256])
	hist3 = cv2.calcHist([figTruth],[2],None,[bins],[0,256])

	magic_hist1 = cv2.calcHist([figEst],[0],None,[bins],[0,256])
	magic_hist2 = cv2.calcHist([figEst],[1],None,[bins],[0,256])
	magic_hist3 = cv2.calcHist([figEst],[2],None,[bins],[0,256])

	gpr_hist1 = cv2.calcHist([figEstGPR],[0],None,[bins],[0,256])
	gpr_hist2 = cv2.calcHist([figEstGPR],[1],None,[bins],[0,256])
	gpr_hist3 = cv2.calcHist([figEstGPR],[2],None,[bins],[0,256])

	true_mean_r = np.zeros((bins, 2))
	true_mean_g = np.zeros((bins, 2))
	true_mean_b = np.zeros((bins, 2))

	magic_mean_r = np.zeros((bins, 2))

	gpr_mean_r = np.zeros((bins, 2))

	for row in range(figTruth.shape[0]):
		for col in range(figTruth.shape[1]):
			r_bin = figTruth[row, col, 2]
			g_bin = figTruth[row, col, 1]
			b_bin = figTruth[row, col, 0]

			true_mean_r[r_bin] += np.array([row, col])

			r_bin = figEst[row, col, 2]
			g_bin = figEst[row, col, 1]
			b_bin = figEst[row, col, 0]

			magic_mean_r[r_bin] += np.array([row, col])

			r_bin = figEstGPR[row, col, 2]
			g_bin = figEstGPR[row, col, 1]
			b_bin = figEstGPR[row, col, 0]

			gpr_mean_r[r_bin] += np.array([row, col])

	true_mean_r /= hist3
	magic_mean_r /= magic_hist3
	gpr_mean_r /= gpr_hist3

	hist1[0] -= n_black_px
	hist2[0] -= n_black_px
	hist3[0] -= n_black_px

	hist1 /= N
	hist2 /= N
	hist3 /= N
	magic_hist1[0] -= n_black_px
	magic_hist2[0] -= n_black_px
	magic_hist3[0] -= n_black_px

	magic_hist1 /= N
	magic_hist2 /= N
	magic_hist3 /= N
	gpr_hist1[0] -= n_black_px
	gpr_hist2[0] -= n_black_px
	gpr_hist3[0] -= n_black_px

	gpr_hist1 /= N
	gpr_hist2 /= N
	gpr_hist3 /= N

	plt.subplot(321, title='Ground truth image'), plt.imshow(figTruth)
	plt.subplot(322, title='Ground truth histogram'),
	plt.plot(hist1, 'b'), plt.plot(hist2, 'g'), plt.plot(hist3, 'r')
	plt.xlim([0,256])
	plt.subplot(323, title='MAGIC-LAT image'), plt.imshow(figEst)
	plt.subplot(324, title='MAGIC-LAT histogram'),
	plt.plot(magic_hist1, 'b'), plt.plot(magic_hist2, 'g'), plt.plot(magic_hist3, 'r')
	plt.xlim([0,256])
	plt.subplot(325, title='GPR image'), plt.imshow(figEstGPR)
	plt.subplot(326, title='GPR histogram'),
	plt.plot(gpr_hist1, 'b'), plt.plot(gpr_hist2, 'g'), plt.plot(gpr_hist3, 'r')
	plt.xlim([0,256])

	plt.tight_layout()
	plt.show()

	true_spatio_r = np.zeros((bins, 3))
	true_spatio_r[:, 0] = hist3.ravel()
	true_spatio_r[:, 1] = true_mean_r[:,0].ravel()
	true_spatio_r[:, 2] = true_mean_r[:,1].ravel()

	magic_spatio_r = np.zeros((bins, 3))
	magic_spatio_r[:, 0] = magic_hist3.ravel()
	magic_spatio_r[:, 1] = magic_mean_r[:,0].ravel()
	magic_spatio_r[:, 2] = magic_mean_r[:,1].ravel()

	gpr_spatio_r = np.zeros((bins, 3))
	gpr_spatio_r[:, 0] = gpr_hist3.ravel()
	gpr_spatio_r[:, 1] = gpr_mean_r[:,0].ravel()
	gpr_spatio_r[:, 2] = gpr_mean_r[:,1].ravel()

	b[0] += calcNMSE(hist1, magic_hist1)
	b[1] += calcNMSE(hist1, gpr_hist1)
	g[0] += calcNMSE(hist2, magic_hist2)
	g[1] += calcNMSE(hist2, gpr_hist2)
	r[0] += calcNMSE(hist3, magic_hist3)
	r[1] += calcNMSE(hist3, gpr_hist3)

	print(calcNMSE(true_spatio_r, magic_spatio_r, multichannel=True))
	print(calcNMSE(true_spatio_r, gpr_spatio_r, multichannel=True))
	exit()

t = len(azim)

with open(os.path.join(outDir, 'metrics_ex.txt'), 'w') as fid:
	fid.write('{:<20}{:<20}{:<20}\n\n'.format('Metric', 'MAGIC-LAT', 'GPR'))
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('NMSE', nmse, nmseGPR))
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('MAE', mae, maeGPR))

	fid.write('\n')
	fid.write('Color-Histogram\n')
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('Avg. NMSE, Blue', b[0]/t, b[1]/t))
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('Avg. NMSE, Green', g[0]/t, g[1]/t))
	fid.write('{:<20}{:<20.6f}{:<20.6f}\n'.format('Avg. NMSE, Red', r[0]/t, r[1]/t))
