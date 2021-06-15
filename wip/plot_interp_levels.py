import os

import numpy as np
import math

# plotting packages
import matplotlib.pyplot as plt


M		=		867		# measured LAT samples (ground truth)
N 		=		6376	# total number of vertices

perc = []
nmse = []
snr = []

with open(os.path.join('.', 'varied_interp_repeated5x.txt'), 'r') as fID:
	for line in fID:
		if line.find('S') != -1:
			continue
		if (line.strip() != ''):
			[v, n, s] = [float(i) for i in line.split(' ') if i != '']
			perc.append(v*100/N)
			nmse.append(n)
			snr.append(s)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10,8))

# axes = ax.flatten()

# ax = axes[0]
ax.plot(perc, nmse, 'o-k')

ax.set_title('Interpolation error versus percentage of input samples', fontsize=18)
ax.set_xlabel(r'Input sample percentage (m/n)', fontsize=16)
ax.set_ylabel('NMSE', fontsize=16)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
plt.grid()

# ax = axes[1]
# ax.plot(perc, snr, 'o-')

# ax.set_title('SNR for various levels of interpolation')
# ax.set_xlabel(r'% interpolation')
# plt.xticks(perc, perc, rotation = 'vertical')
# ax.set_ylabel('SNR')

plt.show()