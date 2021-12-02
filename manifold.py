def isomap(x, n_components, n_neighbors=None, epsilon=None, dist_func=None, cmds_func=None):
	assert(cmds_func is not None)
  	assert((epsilon is not None) or (n_neighbors is not None))
  
  	n_points = x.shape[1]
	
	# Step 1.
  	x_t = x.T
  	nbrs_dist, nbrs_idx = dist_func(x, n_neighbors)
  	D = np.zeros((x_t.shape[0], x_t.shape[0]))
  	enum = enumerate(zip(nbrs_idx, nbrs_dist))
  	for i, elem in enum:
		idxs = elem[0]
		dist = elem[1]
		for j, idx in enumerate(idxs):
			D[i, idx] = dist[j]
    
	# Step 2.
	from scipy.sparse import csr_matrix
	from scipy.sparse.csgraph import shortest_path
	G = csr_matrix(D)
	dist_mat, predecessors = shortest_path(csgraph=G, directed=False, return_predecessors=True)
	
	# Step 3.
	Y, _, _ = cmds(dist_mat, n_compnents, input_type="distance")
	
	return Y, dist_mat, predecessors

# 函数定义部分完毕，下方为瑞士卷绘制示例
from sklearn.datasets import make_swiss_roll

n_points = 1000
data_s_roll, color = make_swiss_roll(n_points)
data_s_roll = data_s_roll.T

fig_swiss_roll = plt.figure()
fig_swiss_roll.suptitle("Swiss roll dataset")

ax = fig_swiss_roll.add_subplot(projection='3d')
ax.scatter(data_s_roll[0, :], data_s_roll[1, :], data_s_roll[2, :], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

# 上方为3D的原图
Y, dist, predecessors = isomap(data_s_roll, n_components=2, n_neighbors=6, epsilon=3.5, dist_func=nearest_neighbor_distance, cmds_func=cmds)
# epsilon在函数中未用到，因使用了k近邻而非固定距离的选取；注意此处的Y为[n, d]的维度，而在assignment4中Y需要转置为[d, n]（其中d=2）
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
# 应可留意到该图为3D瑞士卷的2D展开，且散点所处位置的颜色与3D瑞士卷的颜色是可以对应的。
