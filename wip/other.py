# fig, ax = plt.subplots(figsize=(16, 8), subplot_kw=dict(projection="3d"))

# triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)
# ax.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
# # ax.plot_trisurf(triang, coordinateMatrix[:,2], color='grey')
# pos = ax.scatter(latVer[:,0], latVer[:,1], latVer[:,2], c='blue', s = 20)
# ax.set_title('Simple Case')

# plt.show()


# fig, ax = plt.subplots(figsize=(16, 8), subplot_kw=dict(projection="3d"))

# triang = mtri.Triangulation(coordinateMatrix[:,0], coordinateMatrix[:,1], triangles=connectivityMatrix)
# ax.plot_trisurf(triang, coordinateMatrix[:,2], color='grey', alpha=0.2)
# # ax.plot_trisurf(triang, coordinateMatrix[:,2], color='grey')
# pos = ax.scatter(coordinateMatrix[:,0], coordinateMatrix[:,1], coordinateMatrix[:,2], c=y, cmap='bwr_r', s = 20)
# ax.set_title(nm)

# cax = fig.add_axes([ax.get_position().x1+0.03,ax.get_position().y0,0.01,ax.get_position().height])
# plt.colorbar(pos, cax=cax) # Similar to fig.colorbar(im, cax = cax)

# plt.show()



# for i in range(len(edges)):
# 	print('Edge: '+str(edges[i])+'\tTriangles: '+str(triangles[i]))


# print(len(latCoords), len(coordinateMatrix), len(connectivityMatrix))
# print(max(latVals), min(latVals))

# print(len(coordinateMatrix))
# print(len(latVer))


random.shuffle(idxs)

keep = 0.95
cut = math.floor(keep*len(idxs))

latTrI = sorted(idxs[0:cut])
latTstI = sorted(idxs[cut:len(idxs)])

# lambdas, U = np.linalg.eig(L)

# E = np.diag(lambdas)

# # S = (I + alpha*E)^(-1)
# alpha = 0.5
# S = np.diagflat(np.power(np.ones((1,N)) + alpha*lambdas, -1))

# Ut = np.transpose(U)
# yhat = np.matmul(Ut, y)
# yhat = np.matmul(S, yhat)
# yhat = np.matmul(U, yhat)


# for verNum in idxs:
# 	latTrI = [i for i in idxs if i != verNum]
# 	latTstI = [verNum]

# 	# plotTrainTestVertices(coordinateMatrix, connectivityMatrix, lat, latTrI, latTstI, nm)

# 	for i in range(N):
# 		if i in latTrI:
# 			y[i] = lat[i]['val']
# 			M_l[i,i] = float(1)
# 		else:
# 			y[i] = 0
# 			M_u[i,i] = float(1)

# 	T = np.linalg.inv(M_l + alpha*M_u + beta*L)

# 	yhat = np.matmul(T, alpha*y)

# 	mse = 0
# 	for i in range(N):
# 		if i in latTstI:
# 			e = (yhat[i] - lat[k]['val']) ** 2
# 			mse = mse + e
# 	mse = mse/len(latTstI)