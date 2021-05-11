"""
--------------------------------------------------------------------------------
Plot varied interpolation level error results for MAGIC-LAT.
--------------------------------------------------------------------------------

Description: Reads the text file results in varied_interp_results.txt and 
plots the NMSE versus percentage of input samples.

File contents:
	V           NMSE            SNR

	87          0.0876          21.16264
	109         0.07872         22.0922
	...

Requirements: os, matplotlib

File: plot_varied_interp.py

Author: Jennifer Hellar
Email: jennifer.hellar@rice.edu
--------------------------------------------------------------------------------
"""

import os

import matplotlib.pyplot as plt


M		=		867		# measured LAT samples (ground truth)
N 		=		6376	# total number of vertices

perc = []
nmse = []
snr = []

with open(os.path.join('.', 'varied_interp_results.txt'), 'r') as fID:
	for line in fID:
		if line.find('V') != -1:
			continue
		if (line.strip() != ''):
			[v, n, s] = [float(i) for i in line.split(' ') if i != '']
			perc.append(v*100/N)
			nmse.append(n)
			snr.append(s)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10,8))

ax.plot(perc, nmse, 'o-k')

ax.set_title('Interpolation error versus percentage of input samples', fontsize=18)
ax.set_xlabel(r'Input sample percentage (m/n)', fontsize=16)
ax.set_ylabel('NMSE', fontsize=16)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
plt.grid()

plt.show()