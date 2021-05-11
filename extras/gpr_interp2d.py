from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

rng = np.random.RandomState(0)


x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)

xx, yy = np.meshgrid(x, y, sparse=True)


# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
z = xx**2 + yy**2

# print(xx)
# print(yy)
# print(z)


X = []
out = []
for i in range(x.shape[0]):
	for j in range(y.shape[0]):
		X.append([x[i], y[j]])
		out.append(x[i]**2 + y[j]**2)


# gp_kernel = ExpSineSquared()
gp_kernel = RBF()
gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)
gpr.fit(X, out)

x_plot = np.arange(-5, 5, 0.1)
y_plot = np.arange(-5, 5, 0.1)

xx_plot, yy_plot = np.meshgrid(x_plot, y_plot, sparse=True)

X_plot = []
for i in range(x_plot.shape[0]):
	for j in range(y_plot.shape[0]):
		X_plot.append([x_plot[i], y_plot[j]])

out_gpr = gpr.predict(X_plot, return_std=False)

# print(out_gpr)
z_out = []
idx = 0
for i in range(x_plot.shape[0]):
	for j in range(y_plot.shape[0]):
		if (j == 0):
			z_out.append([])
		z_out[i].append(out_gpr[idx])
		idx += 1

z_out = np.array(z_out)

z_true = xx_plot**2 + yy_plot**2

err = abs(z_out - z_true)
M = x_plot.shape[0]*y_plot.shape[0]

mse = 1/M*np.sum(err)
print(mse)

snr = 20*np.log10(np.sum(z_true ** 2)/(M*mse))
print(snr)

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(16, 8), subplot_kw=dict(projection="3d"))
axes = ax.flatten()

thisAx = axes[0]
thisAx.plot_surface(xx, yy, z,cmap='viridis', edgecolor='none')
thisAx.set_title('Training')

thisAx = axes[1]
thisAx.plot_surface(xx_plot, yy_plot, z_out,cmap='viridis', edgecolor='none')
thisAx.set_title('Interpolated')

thisAx = axes[2]
thisAx.plot_surface(xx_plot, yy_plot, z_true, cmap='viridis', edgecolor='none')
thisAx.set_title('True')
	
plt.show()