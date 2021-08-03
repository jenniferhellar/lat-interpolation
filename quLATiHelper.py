
import numpy as np
import os

from qulati import gpmi, eigensolver


def quLATiModel(patient, vertices, faces):
	QFile = 'Q_p{}.npy'.format(patient)
	VFile = 'V_p{}.npy'.format(patient)
	gradVFile = 'gradV_p{}.npy'.format(patient)

	# model with reduced rank efficiency
	if not os.path.isfile(QFile):
		Q, V, gradV, centroids = eigensolver(vertices, np.array(faces), holes = 0, layers = 10, num = 256)
		with open(QFile, 'wb') as fid:
			np.save(fid, Q)
		with open(VFile, 'wb') as fid:
			np.save(fid, V)
		with open(gradVFile, 'wb') as fid:
			np.save(fid, gradV)
	else:
		Q = np.load(QFile, allow_pickle = True)
		V = np.load(VFile, allow_pickle = True)
		gradV = np.load(gradVFile, allow_pickle = True)

	model = gpmi.Matern(vertices, np.array(faces), Q, V, gradV, JAX = False)

	return model	


def quLATi(TrIdx, TrVal, vertices, model):
	obs = np.array(TrVal)
	trVertices = np.array(TrIdx)

	model.set_data(obs, trVertices)
	model.kernelSetup(smoothness = 3./2.)

	# optimize the nugget
	model.optimize(nugget = None, restarts = 5)

	pred_mean, pred_stdev = model.posterior(pointwise = True)

	est = pred_mean[0:vertices.shape[0]]

	return est