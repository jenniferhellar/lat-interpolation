import os

import numpy as np

# plotting packages
import matplotlib.pyplot as plt



def plot_single(fileDir, patient):

	meanFile = os.path.join(fileDir, 'p{}_mean.txt'.format(patient))
	stdFile = os.path.join(fileDir, 'p{}_std.txt'.format(patient))

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
			m.append(float(lineSplit[0]))
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

	leg = ['MAGIC-LAT', 'GPR', 'quLATi']

	# beta on the x-axis, one line per alpha value
	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16,8))
	ax.plot(m, magicMean, 'o-')
	ax.plot(m, gprMean, 'o-')
	ax.plot(m, quLATiMean, 'o-')

	# ax.errorbar(m, magicMean, yerr = magicStd, capsize = 4)
	# ax.scatter(m, magicMean)

	# ax.errorbar(m, gprMean, yerr = gprStd, capsize = 4)
	# ax.scatter(m, gprMean)

	# ax.errorbar(m, quLATiMean, yerr = quLATiStd, capsize = 4)
	# ax.scatter(m, quLATiMean)

	ax.set_title(r'$\Delta$E* vs m, Patient ' + patient)
	ax.set_xlabel('m')
	plt.xticks(m, m, rotation = 'vertical')
	ax.set_ylabel(r'$\Delta$E*')
	ax.legend(leg)

	plt.show()

fileDir = 'test_varied_m_results'
patient = '033'

plot_single(fileDir, patient)