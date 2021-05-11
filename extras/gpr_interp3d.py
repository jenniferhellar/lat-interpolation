from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

rng = np.random.RandomState(0)


x = np.linspace(-5, 5, 5)
y = np.linspace(-5, 5, 5)
z = np.linspace(-5, 5, 5)

xx, yy, zz = np.meshgrid(x, y, z, sparse=True)


f = xx**2 + yy**2 + zz**2


X = []
out = []
for i in range(x.shape[0]):
	for j in range(y.shape[0]):
		for k in range(z.shape[0]):
			X.append([x[i], y[j], z[k]])
			out.append(x[i]**2 + y[j]**2 + z[k]**2)
X = np.array(X)


# gp_kernel = ExpSineSquared()
gp_kernel = RBF()
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)
gpr.fit(X, out)

x_plot = np.arange(-5, 5, 0.1)
y_plot = np.arange(-5, 5, 0.1)
z_plot = np.arange(-5, 5, 0.1)

xx_plot, yy_plot, zz_plot = np.meshgrid(x_plot, y_plot, z_plot, sparse=True)

X_plot = []
for i in range(x_plot.shape[0]):
	for j in range(y_plot.shape[0]):
		for k in range(z_plot.shape[0]):
			X_plot.append([x_plot[i], y_plot[j], z_plot[k]])
X_plot = np.array(X_plot)

out_gpr = gpr.predict(X_plot, return_std=False)

# print(out_gpr)
f_out = np.zeros((x_plot.shape[0], y_plot.shape[0], z_plot.shape[0]))
idx = 0
for i in range(x_plot.shape[0]):
	for j in range(y_plot.shape[0]):
		for k in range(z_plot.shape[0]):
			f_out[i][j][k] = out_gpr[idx]
			idx += 1

f_true = xx_plot**2 + yy_plot**2 + zz_plot**2

err = abs(f_out - f_true)
M = x_plot.shape[0]*y_plot.shape[0]*z_plot.shape[0]

mse = 1/M*np.sum(err)
print(mse)

snr = 20*np.log10(np.sum(f_true ** 2)/(M*mse))
print(snr)

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()

thisAx = axes[0]
thisAx.scatter(X[:,0], X[:,1], X[:,2], c=f, cmap='viridis', edgecolor='none')
thisAx.set_title('Training')

thisAx = axes[1]
thisAx.scatter(X_plot[:,0], X_plot[:,1], X_plot[:,2], c=f_out,cmap='viridis', edgecolor='none')
thisAx.set_title('Interpolated')

thisAx = axes[2]
thisAx.scatter(X_plot[:,0], X_plot[:,1], X_plot[:,2], c=f_true, cmap='viridis', edgecolor='none')
thisAx.set_title('True')
	
plt.show()