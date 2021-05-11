import os

import numpy as np
import math

# plotting packages
import matplotlib.pyplot as plt


alphas = []
betas = []

with open(os.path.join('res_lows', 'alphas.txt'), 'r') as fID:
	for line in fID:
		line = line.strip()
		if (line != ''):
			alphas.append(float(line))

with open(os.path.join('res_lows', 'betas.txt'), 'r') as fID:
	for line in fID:
		line = line.strip()
		if (line != ''):
			betas.append(float(line))


nmse = []
for i in range(len(alphas)):
	nmse.append([])
i = 0
with open(os.path.join('res_lows', 'nmse.txt'), 'r') as fID:
	for line in fID:
		line = line.strip()
		if (line != ''):
			a_idx = math.floor(i / len(betas))
			nmse[a_idx].append(float(line))
			i += 1

with open(os.path.join('res_alpha_high', 'alphas.txt'), 'r') as fID:
	for line in fID:
		line = line.strip()
		if (line != '') and (float(line) not in alphas):
			alphas.append(float(line))

with open(os.path.join('res_alpha_high', 'betas.txt'), 'r') as fID:
	for line in fID:
		line = line.strip()
		if (line != '') and (float(line) not in betas):
			betas.append(float(line))

i = 0

old_alphas = len(nmse)
while len(nmse) < len(alphas):
	nmse.append([])

with open(os.path.join('res_alpha_high', 'nmse.txt'), 'r') as fID:
	for line in fID:
		line = line.strip()
		if (line != ''):
			a_idx = old_alphas + math.floor(i / len(betas))
			nmse[a_idx].append(float(line))
			i += 1

i = 0
a_idx = -1

old_betas = len(nmse[0])

with open(os.path.join('res_beta_high', 'nmse.txt'), 'r') as fID:
	for line in fID:
		line = line.strip()
		if (line != ''):
			if (i % (len(betas) - old_betas) == 0):
				a_idx += 1
			nmse[a_idx].append(float(line))
			i += 1

# print(alphas)
# print(nmse)
nmse = np.array(nmse)
nmse = np.transpose(nmse)

leg = []

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16,8))
for i in range(len(alphas)):
	ax.plot(alphas, nmse[i], 'o-')

for i in range(len(betas)):
	leg.append(r'$\beta$ = ' + str(betas[i]))

ax.set_title('Regularization Parameter Cross-Validation (coarse)')
ax.set_xlabel(r'$\alpha$')
plt.xticks(alphas, alphas, rotation = 'vertical')
plt.xscale('log')
ax.set_ylabel('NMSE')
ax.legend(leg)

plt.ylim(0, 1.3)
plt.hlines(0.11, xmin=0, xmax=0.1, linestyles='dashed')
plt.vlines(0.1, ymin=0, ymax=0.11, linestyles='dashed')
plt.text(0.15, 0.08, r'Minimum NMSE: $\alpha \leq 0.1$, $\beta \geq 0.1$')

# plt.show()


alphas = []
betas = []

with open(os.path.join('res_fine', 'alphas.txt'), 'r') as fID:
	for line in fID:
		line = line.strip()
		if (line != ''):
			alphas.append(float(line))

with open(os.path.join('res_fine', 'betas.txt'), 'r') as fID:
	for line in fID:
		line = line.strip()
		if (line != ''):
			betas.append(float(line))


nmse = []
for i in range(len(alphas)):
	nmse.append([])
i = 0
with open(os.path.join('res_fine', 'nmse.txt'), 'r') as fID:
	for line in fID:
		line = line.strip()
		if (line != ''):
			a_idx = math.floor(i / len(betas))
			nmse[a_idx].append(float(line))
			i += 1


nmse = np.array(nmse)
nmse = np.transpose(nmse)

leg = []

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16,8))
for i in range(1, len(betas)):
	ax.plot(alphas, nmse[i], 'o-')

for i in range(1, len(betas)):
	leg.append(r'$\beta$ = ' + str(betas[i]))

ax.set_title('Regularization Parameter Cross-Validation (fine)')
ax.set_xlabel(r'$\alpha$')
plt.xticks(alphas, alphas, rotation = 'vertical')
plt.xscale('log')
ax.set_ylabel('NMSE')
ax.legend(leg)

# plt.ylim(0, 1.3)
# plt.hlines(0.11, xmin=0, xmax=0.1, linestyles='dashed')
# plt.vlines(0.1, ymin=0, ymax=0.11, linestyles='dashed')
# plt.text(0.15, 0.08, r'Minimum NMSE: $\alpha \leq 0.1$, $\beta \geq 0.1$')

plt.show()