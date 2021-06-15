from pygsp import graphs, filters, plotting, utils
import matplotlib.pyplot as plt
import numpy as np
import random

G = graphs.Bunny()
G.estimate_lmax()

s = np.zeros(G.N)
DELTAS = [20, 30, 1090]
s[DELTAS] = 1
g = filters.Heat(G, tau=100)
s = g.filter(s)
G.plot_signal(s, backend='matplotlib')


allIdx = [i for i in range(G.N)]
TstIdx = [i for i in allIdx if (i % 2) == 0]
TrIdx = [i for i in allIdx if i not in TstIdx]

ssub = np.array([i for i in s[:]])
ssub[TstIdx] = 0
G.plot_signal(ssub)


g = filters.Abspline(G, Nf=3)

g1 = filters.Abspline(G, Nf=2)

# fig, ax = plt.subplots(figsize=(10, 5))
# g.plot(ax=ax)
# _ = ax.set_title('Filter bank of Abspline wavelets')

# DELTA = 20
# sloc = g.localize(DELTA)

# fig = plt.figure(figsize=(10, 2.5))
# for i in range(3):
#      ax = fig.add_subplot(1, 3, i+1, projection='3d')
#      G.plot_signal(sloc[:, i], ax=ax)
#      _ = ax.set_title('Wavelet {}'.format(i+1))
#      ax.set_axis_off()
# fig.tight_layout()

sf = g.filter(ssub)

G.plot_signal(sf[:,0], backend='matplotlib')
G.plot_signal(sf[:,1], backend='matplotlib')
G.plot_signal(sf[:,2], backend='matplotlib')

s_out = g1.filter(sf[:,0:2])

G.plot_signal(s_out)

est = s_out[TstIdx]
true = s[TstIdx]
nmse = np.sum(abs(est - true)**2)/np.sum((true - np.mean(true))**2)
print(nmse)

plt.show()