"""
--------------------------------------------------------------------------------
Plots results for cross-validation over multiple input sizes..
--------------------------------------------------------------------------------

Description: Plots results of test_varied_m.py

Requirements: os, math, matplotlib

File: plot_varied_m.py

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""
import os

# plotting packages
import matplotlib.pyplot as plt

import math



def plot_single(fileDir, patient, id, title):

	meanFile = os.path.join(fileDir, 'p' + patient + '_' + id + '_mean.txt')
	stdFile = os.path.join(fileDir, 'p' + patient + '_' + id + '_std.txt')

	m = []

	magicMean = []
	gprMean = []
	quLATiMean = []

	magicStd = []
	gprStd = []
	quLATiStd = []

	with open(meanFile, 'r') as fID:
		for line in fID:
			if line[0] == 'm':
				continue
			lineSplit = line.split(' ')
			lineSplit = [i.strip() for i in lineSplit if i.strip() != '']
			m.append(int(lineSplit[0]))
			magicMean.append(float(lineSplit[1]))
			gprMean.append(float(lineSplit[2]))
			quLATiMean.append(float(lineSplit[3]))

	with open(stdFile, 'r') as fID:
		for line in fID:
			if line[0] == 'm':
				continue
			lineSplit = line.split(' ')
			lineSplit = [i.strip() for i in lineSplit if i.strip() != '']
			magicStd.append(2*float(lineSplit[1]))
			gprStd.append(2*float(lineSplit[2]))
			quLATiStd.append(2*float(lineSplit[3]))

	leg = ['GPR', 'GPMI', 'MAGIC-LAT']

	# beta on the x-axis, one line per alpha value
	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(14,22))
	ax.plot(m[1:], gprMean[1:], 's-', linewidth=5, ms=15)
	ax.plot(m[1:], quLATiMean[1:], '^-', linewidth=5, ms=15)
	ax.plot(m[1:], magicMean[1:], 'D-', linewidth=5, ms=15)

	# ax.errorbar(m, magicMean, yerr = magicStd, capsize = 4)
	# ax.scatter(m, magicMean)

	# ax.errorbar(m, gprMean, yerr = gprStd, capsize = 4)
	# ax.scatter(m, gprMean)

	# ax.errorbar(m, quLATiMean, yerr = quLATiStd, capsize = 4)
	# ax.scatter(m, quLATiMean)

	ax.grid(True)
	ax.set_title(title, size='36')
	ax.set_xlabel('Number of LAT observations (m)', size=30)
	plt.xticks(m[1:], m[1:], rotation = 'vertical', size=24)

	ymax = math.ceil(max(max(gprMean), max(quLATiMean), max(magicMean))/5)*5
	if patient == '035':
		ymax = 10
	plt.ylim((0, ymax))
	plt.yticks(size=24)
	ax.set_ylabel('Mean Delta-E (MDE)', size=30)
	ax.legend(leg, fontsize=30, loc='lower right')

	plt.show()

fileDir = 'test_varied_m_results'

PATIENT_IDX = 11

if PATIENT_IDX == 4:
	patient, id, title = ('033', '3', 'Map 0 (Patient A)')
elif PATIENT_IDX == 5:
	patient, id, title = ('033', '4', 'Map 1 (Patient A)')
elif PATIENT_IDX == 6:
	patient, id, title = ('034', '4', 'Map 2 (Patient B)')
elif PATIENT_IDX == 7:
	patient, id, title = ('034', '5', 'Map 3 (Patient B)')
elif PATIENT_IDX == 8:
	patient, id, title = ('034', '6', 'Map 4 (Patient B)')
elif PATIENT_IDX == 9:
	patient, id, title = ('035', '8', 'Map 5 (Patient C)')
elif PATIENT_IDX == 11:
	patient, id, title = ('037', '9', 'Map 6 (Patient D)')


plot_single(fileDir, patient, id, title)