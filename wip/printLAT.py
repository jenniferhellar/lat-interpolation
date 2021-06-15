"""
--------------------------------------------------------------------------------
Manifold Approximating Graph Interpolation on Cardiac mapLAT data (MAGIC-mapLAT).
--------------------------------------------------------------------------------

Description: Cross-validation to randomly select test sets for interpolation.  
5x repetitition for error mean and variance estimation.

Requirements: os, numpy, matplotlib, sklearn, scipy, math

File: gpr_interp.py

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""

import time

import os

import numpy as np
import math
import random

# plotting packages
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import cv2

# functions to read the files
from readMesh import readMesh
from readLAT import readLAT


from utils import *
from const import *
from magicLAT import *


# 20 is the one I have been working with
PATIENT_MAP				=		20

NUM_TRAIN_SAMPS 		= 		840
TEST_CHUNK_SIZE			=		20
NUM_TEST_REPEATS 		= 		2
EDGE_THRESHOLD			=		50

""" Read the files """
meshFile = meshNames[PATIENT_MAP]
latFile = latNames[PATIENT_MAP]
nm = meshFile[0:-5]
patient = nm[7:10]

print('Reading files for ' + nm + ' ...\n')
[coordinateMatrix, connectivityMatrix] = readMesh(dataDir + meshFile)
[latCoords, latVals] = readLAT(dataDir + latFile)

n = len(coordinateMatrix)

mapIdx = [i for i in range(n)]
mapCoord = [coordinateMatrix[i] for i in mapIdx]

latIdx, latCoords, latVals = mapSamps(mapIdx, mapCoord, latCoords, latVals)

fid = open(os.path.join('results','p{}_LAT.txt'.format(patient)), 'w')
np.array(latVals).tofile(fid, sep='\n', format='%.5f')
fid.close()