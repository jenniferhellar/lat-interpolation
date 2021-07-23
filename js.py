import numpy as np
import math

def js(mu1, mu2, sigma1, sigma2, n1, n2):
	if (n1 == 0 and n2 == 0):
		return 0

	mu_hat = (n2*mu1 + n1*mu2)/(n1 + n2)

	sigma_hat = n2/(n1+n2) * (sigma1 + np.matmul(mu1, mu1.T)) + \
		n1/(n1+n2) * (sigma2 + np.matmul(mu2, mu2.T)) - np.matmul(mu_hat, mu_hat.T)

	if np.linalg.det(sigma1) <= 0 or np.linalg.det(sigma2) <= 0 or np.linalg.det(sigma_hat) <= 0:
		return 0

	t0 = math.log(np.linalg.det(sigma_hat))

	t1 = 1/2 * np.trace(np.matmul(np.linalg.inv(sigma_hat), (sigma1 + sigma2)))

	t2 = 1/4 * np.matmul((mu1 - mu2).T, np.matmul(np.linalg.inv(sigma_hat), (mu1 - mu2)))

	t3 = 1/2 * math.log(np.linalg.det(sigma1) * np.linalg.det(sigma2))

	js_div = t0 + t1 - 2 + t2 - t3

	return js_div