from preprocessing import preprocess
from mu import mu
from als import *
import numpy as np
from sklearn.decomposition import NMF


def printClusters(W, features, cluster_size):
	'''
	Print clusters found on W.
	
	Parameters:
		W: ndarray
			- a matrix containing clusters found by non-negative matrix factorization
		features: list
			- a list of features
		cluster_size: int
			- an integer specifying the number of features in a cluster

	Returns:
		None
	'''
	print('---------------------------------------------------------------------')
	print('Discovered clusters')
	print('')

	sorted_W = W[:, np.argsort(W.sum(axis = 0))]

	for i in range(np.size(W, 1)):
		cluster = []
		idx = (-sorted_W[:,i]).argsort()[:cluster_size]
		for j in range(np.size(idx)):
			cluster.append(features[idx[j]])
		print('Cluster ' + str(i + 1) + ': ' + ', '.join(cluster))

	print('')


if __name__ == '__main__':
	# Set variables
	filename = 'abcnews-date-text.csv'
	#filename = 'USvideos.csv'
	col_name = 'headline_text'
	#col_name = 'title'
	data_frac = 0.01
	random_sample = True
	num_max_feature = 1000
	num_clusters = 5
	num_iter = 10
	cluster_size = 10
	method = 'all'

	# Preprocess the dataset
	A, features = preprocess(filename = filename, col_name = col_name, data_frac = data_frac, random_sample = random_sample, num_max_feature = num_max_feature)

	# Run a desired algorithm to perform non-negative matrix factorization on A
	if method == 'all':
		# Initialize W and H
		# Use the same matrices for all the algorithms for comparison
		init_W = np.random.rand(np.size(A, 0), num_clusters)
		init_H = np.random.rand(num_clusters, np.size(A, 1))

		# Run multiplicative updates with init_W and init_H
		W, H = mu(A, num_clusters, delta = 0.0000001, num_iter = num_iter, init_W = init_W, init_H = init_H)
		printClusters(W, features, cluster_size)

		# Run alternating least squares with init_W and init_H
		W, H = als(A, num_clusters, num_iter = num_iter, method = 'als', init_W = init_W, init_H = init_H)
		printClusters(W, features, cluster_size)

		# Run alternating non-negative least squares with active set with init_W and init_H
		W, H = als(A, num_clusters, num_iter = num_iter, method = 'anls_as', init_W = init_W, init_H = init_H)
		printClusters(W, features, cluster_size)

	elif method == 'mu':
		W, H = mu(A, num_clusters, delta = 0.0000001, num_iter = num_iter)

	elif method == 'als':
		W, H = als(A, num_clusters, num_iter = num_iter, method = 'als')

	elif method == 'anls':
		W, H = als(A, num_clusters, num_iter = num_iter, method = 'anls_as')

	elif method == 'sklearn':
		model = NMF(n_components = num_clusters, init='nndsvd')
		W = model.fit_transform(A)
		H = model.components_

	# Print clusters found by non-negative matrix factorization
	if method != 'all':
		printClusters(W, features, cluster_size)