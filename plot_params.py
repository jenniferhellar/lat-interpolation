import os

# plotting packages
import matplotlib.pyplot as plt

fileDir = 'params_results'
fileName = 'p033_t50_m100_r50_de2000.txt'
# fileName = 'p034_t50_m100_r20_de2000.txt'
# fileName = 'p035_t50_m100_r20_de2000.txt'
# fileName = 'p037_t50_m100_r20_de2000.txt'

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
	ax.plot(betas, res[alphas[i]], 'o-')
	ax.errorbar(betas, res[alphas[i]], yerr = double_std[alphas[i]], 
            linewidth = 1.5, color = "black", alpha = 0.4, capsize = 4)

for i in range(len(alphas)):
	leg.append(r'$\alpha$ = ' + str(alphas[i]))

ax.set_title('Regularization Parameter Cross-Validation, Patient ' + fileName[1:4])
ax.set_xlabel(r'$\beta$')
plt.xticks(alphas, alphas, rotation = 'vertical')
plt.xscale('log')
ax.set_ylabel(r'$\Delta$E*')
ax.legend(leg)

plt.show()