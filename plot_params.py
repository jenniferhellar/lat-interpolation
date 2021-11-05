"""
--------------------------------------------------------------------------------
Plots results for cross-validation over regularization parameters.
--------------------------------------------------------------------------------

Description: Plots output of params.py

Requirements: os, numpy, matplotlib

File: plot_params.py

Author: Jennifer Hellar
Email: jenniferhellar@gmail.com
--------------------------------------------------------------------------------
"""

import os
import numpy as np

# plotting packages
import matplotlib.pyplot as plt



def plot_single(fileDir, fileName):
	""" Plots result for a single patient/map. """
	alphas = []
	betas = []

	res = {}
	double_std = {}

	cnt = 0

	with open(os.path.join(fileDir, fileName), 'r') as fID:
		for line in fID:
			if line[0] == 'a':
				continue
			lineSplit = line.split(' ')
			lineSplit = [i.strip() for i in lineSplit if i.strip() != '']
			alpha = float(lineSplit[0])
			beta = float(lineSplit[1])
			mean = float(lineSplit[2])
			std = float(lineSplit[3])

			if alpha not in res.keys():
				res[alpha] = [mean]
				double_std[alpha] = [2*std]
				alphas.append(alpha)
				cnt += 1
			else:
				res[alpha].append(mean)
				double_std[alpha].append(2*std)
			if cnt < 2:
				betas.append(beta)

	leg = []

	# beta on the x-axis, one line per alpha value
	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16,8))
	for i in range(len(alphas)):
		# ax.plot(betas, res[alphas[i]], 'o-')
		ax.errorbar(betas, res[alphas[i]], yerr = double_std[alphas[i]], capsize = 4)
		ax.scatter(betas, res[alphas[i]])

	for i in range(len(alphas)):
		leg.append(r'$\alpha$ = ' + str(alphas[i]))

	ax.set_title('Regularization Parameter Cross-Validation, Patient ' + fileName[1:4])
	ax.set_xlabel(r'$\beta$')
	plt.xticks(alphas, alphas, rotation = 'vertical')
	plt.xscale('log')
	ax.set_ylabel(r'$\Delta$E*')
	ax.legend(leg)

	plt.show()



def plot_average(fileDir, fileNames):
	""" Plots average across patients. """

	alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
	betas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

	res = np.zeros((7,7))
	double_std = np.zeros((7,7))

	cnt = 0

	for fileName in fileNames:
		with open(os.path.join(fileDir, fileName), 'r') as fID:
			for line in fID:
				if line[0] == 'a':
					continue
				lineSplit = line.split(' ')
				lineSplit = [i.strip() for i in lineSplit if i.strip() != '']
				alpha = float(lineSplit[0])
				beta = float(lineSplit[1])
				mean = float(lineSplit[2])
				std = float(lineSplit[3])

				a = alphas.index(alpha)
				b = betas.index(beta)

				res[a, b] += mean
				double_std[a, b] += std

	res = 1/len(fileNames) * res
	double_std = 1/len(fileNames) * double_std

	leg = []

	# beta on the x-axis, one line per alpha value
	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16,8))
	for i in range(len(alphas)):
		# ax.plot(betas, res[i], 'o-')
		ax.errorbar(betas, res[i], yerr = double_std[i], capsize = 4)
		ax.scatter(betas, res[i])

	for i in range(len(alphas)):
		leg.append(r'$\alpha$ = ' + str(alphas[i]))

	ax.set_title('Regularization Parameter Cross-Validation, Avg Across Patients')
	ax.set_xlabel(r'$\beta$')
	plt.xticks(betas, betas, rotation = 'vertical')
	plt.xscale('log')
	ax.set_ylabel(r'$\Delta$E*')
	ax.legend(leg)

	plt.show()


fileDir = 'params_results'
fileName = 'p033_t50_m100_r20_de2000.txt'
# fileName = 'p034_t50_m100_r20_de2000.txt'
# fileName = 'p035_t50_m100_r20_de2000.txt'
# fileName = 'p037_t50_m100_r20_de2000.txt'

fileNames = ['p033_t50_m100_r20_de2000.txt', 'p034_t50_m100_r20_de2000.txt',
	'p035_t50_m100_r20_de2000.txt', 'p037_t50_m100_r20_de2000.txt']

plot_single(fileDir, fileName)
plot_average(fileDir, fileNames)