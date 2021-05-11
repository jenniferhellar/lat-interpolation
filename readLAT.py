
"""
--------------------------------------------------------------------------------
Text reader for CARTO exported LAT spatial data.
--------------------------------------------------------------------------------
Description: Reads LAT values from the CARTO LATSpatialData.txt file and
returns a numpy array of sample coordinates and a list of corresponding
LAT values.

File contents:
	Point Name,X,Y,Z,LAT
	P2,-18.9582,67.7725,134.415,-10000
	P3,-25.669,75.1242,137.267,-10000
	P4,-29.8143,84.4753,139.026,-10000
	P5,-8.82908,54.6819,129.527,-10000
	P6,-17.1478,59.3911,133.56,-10000
	P7,-23.8857,66.7642,136.372,-10000
	P8,-28.438,75.9833,137.922,-10000
	P9,3.34371,58.3584,123.113,-10000
	P10,-8.17534,59.325,129.387,-10000
	P11,-16.6266,63.4593,133.674,-10000
	P12,-23.3143,70.7666,136.591,-10000
	P13,-27.3994,80.1868,137.933,-10000
	P14,23.0643,46.2865,127.094,-31
	P15,12.391,53.0813,131.01,-26

Requirements: numpy

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""
import numpy as np

def readLAT(fileName):
	latCoords = []
	latVals = []

	with open(fileName,'r') as fID:
		for line in fID:
			line = line.strip()
			if (line[0:5] == 'Point'):
				continue
			else:
				if (line != ''):
					[X,Y,Z,lat] = map(float,line.split(',')[1:])
					if (lat != -10000):
						latCoords.append([X,Y,Z])
						latVals.append(lat)
	return [np.array(latCoords), latVals]