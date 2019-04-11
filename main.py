from preprocessing import preprocess
from mu import mu
from als import *
import numpy as np
from sklearn.decomposition import NMF


def printClusters(W, features, cluster_size):
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
	data_frac = 0.01
	num_max_feature = 1000
	num_clusters = 5
	num_iter = 10
	cluster_size = 10
	method = 'anls'

	# Preprocess the dataset
	A, features = preprocess(filename = 'abcnews-date-text.csv', col_name = 'headline_text', data_frac = data_frac, random_sample = False, num_max_feature = num_max_feature)

	# Run a desired algorithm to perform non-negative matrix factorization on A
	if method == 'mu':
		W, H = mu(A, num_clusters, delta = 0.0000001, num_iter = num_iter)
	elif method == 'als':
		W, H = als(A, num_clusters, num_iter = num_iter, method = 'als')
	elif method == 'anls':
		W, H = als(A, num_clusters, num_iter = num_iter, method = 'anls_as')
	elif method == 'sklearn':
		model = NMF(n_components = num_clusters, init='nndsvd')
		W = model.fit_transform(A)
		H = model.components_

	printClusters(W, features, cluster_size)