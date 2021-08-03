
"""
Requirements: numpy, scipy, matplotlib, scikit-learn
"""

import os

# plotting packages
from vedo import *

# loading images
import cv2

from utils import *
from calcVisualMetrics import *


outDir				 	=		'results_toy'
if not os.path.isdir(outDir):
	os.makedirs(outDir)

gridX = 10
gridY = 10
mesh = Grid(pos=(0,0,0), sx=gridX, sy=gridY, c='black')

vertices = mesh.vertices()
faces = mesh.faces()

mesh = Mesh([vertices, faces], c='black')

trueVals = []

for [x,y,z] in vertices:
	if x < 0:
		trueVals.append(-100)
	else:
		trueVals.append(100)

# mesh.backColor('violet').lineColor('tomato').lineWidth(2)

vplt = Plotter(N=1, axes=0, offscreen=True)
truePts = Points(vertices, r = 20).cmap('rainbow_r', trueVals, vmin=-100, vmax=100)
vplt.show(mesh, truePts, bg='black')
vplt.screenshot(filename=os.path.join(outDir, 'true.png'), returnNumpy=False)
vplt.close()

trueNMSE = calcNMSE(trueVals, trueVals)
trueMAE = calcMAE(trueVals, trueVals)
(trueCorr, trueSpatio) = calcVisualMetrics(outDir, 'true.png', 'true.png')

greenLineVals = []
for [x,y,z] in vertices:
	if x < 0:
		greenLineVals.append(-100)
	elif x == 0.0:
		greenLineVals.append(0)
	else:
		greenLineVals.append(100)


vplt = Plotter(N=1, axes=0, offscreen=True)
newGreenPts = Points(vertices, r = 20).cmap('rainbow_r', greenLineVals, vmin=-100, vmax=100)
vplt.show(mesh, newGreenPts, bg='black')
vplt.screenshot(filename=os.path.join(outDir, 'greenLine.png'), returnNumpy=False)
vplt.close()

greenLineNMSE = calcNMSE(trueVals, greenLineVals)
greenLineMAE = calcMAE(trueVals, greenLineVals)
(greenLineCorr, greenLineSpatio) = calcVisualMetrics(outDir, 'true.png', 'greenLine.png', idx=1)


shiftLineVals = []
for [x,y,z] in vertices:
	if x < 1:
		shiftLineVals.append(-100)
	else:
		shiftLineVals.append(100)


vplt = Plotter(N=1, axes=0, offscreen=True)
shiftLinePts = Points(vertices, r = 20).cmap('rainbow_r', shiftLineVals, vmin=-100, vmax=100)
vplt.show(mesh, shiftLinePts, bg='black')
vplt.screenshot(filename=os.path.join(outDir, 'shiftLine.png'), returnNumpy=False)
vplt.close()

shiftLineNMSE = calcNMSE(trueVals, shiftLineVals)
shiftLineMAE = calcMAE(trueVals, shiftLineVals)
(shiftLineCorr, shiftLineSpatio) = calcVisualMetrics(outDir, 'true.png', 'shiftLine.png', idx=2)

with open(os.path.join(outDir, 'metrics.txt'), 'w') as fid:
	fid.write('{:>65}\n'.format('Metric'))
	fid.write('{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}\n'.format('', '', 'NMSE', 'MAE', 'Histogram Corr.', 'Spatiogram Corr.'))
	fid.write('{:<10}{:<20}{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('Image', 'true.png', 
		trueNMSE, trueMAE, trueCorr, trueSpatio))
	fid.write('{:<10}{:<20}{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('', 'greenLine.png', 
		greenLineNMSE, greenLineMAE, greenLineCorr, greenLineSpatio))
	fid.write('{:<10}{:<20}{:<20.6f}{:<20.6f}{:<20.6f}{:<20.6f}\n'.format('', 'shiftLine.png', 
		shiftLineNMSE, shiftLineMAE, shiftLineCorr, shiftLineSpatio))
