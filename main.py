import sys
import argparse

import numpy as np
from sklearn.decomposition import NMF

from preprocessing import preprocess
from preprocessing_sport_data import preprocess_bbcsport
from mu import mu
from als import *


def print_clusters(W, features, cluster_size):
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


def process_arg():
	'''
	Process the user input.

	Parameters:
		None

	Returns:
		args: argparse.Namespace()
	'''
	parser = argparse.ArgumentParser(description = "This program applies a nonnegative matrix factorization algorithms to a dataset for clustering.")
	parser.add_argument('-f', '--filename', type = str, required = True,  help = 'the input file name')
	parser.add_argument('-c', '--col_name', type = str, required = True,  help = 'the column of the input csv file for nonnegative matrix factorization.')
	parser.add_argument('-m', '--method', choices = {'all', 'mu', 'als', 'anls', 'sklearn'}, type = str, required = True,  help = 'the NMF method to apply')
	parser.add_argument('-d', '--data_frac', default = 1, type = float, required = False, help = 'the amount of the data to be used')
	parser.add_argument('-r', '--random_sample', default = True, type = bool, required = False,  help = 'if set False, disables random sampling of the data')
	parser.add_argument('-n', '--num_max_feature', default = 1000, type = float, required = False,  help = 'the maximum number of features to be discovered in the dataset')
	parser.add_argument('-s', '--cluster_size', default = 10, type = float, required = False,  help = 'the number of features in each cluster')
	parser.add_argument('-k', '--num_clusters', default = 5, type = float, required = False,  help = 'the number of clusters to be discovered')
	parser.add_argument('-i', '--num_iters', default = 100, type = float, required = False,  help = 'the number of iterations to run a NMF algorithm')
	parser.add_argument('-p', '--print_enabled', default = False, type = bool, required = False,  help = 'if ture, output print statements')

	args = parser.parse_args()

	return args


def process_NMF(args, A, features):
	'''
	Using a specified algorithm by user, apply nonnegative matrix factorization to the given matrix A.
	Based on the factorization, print clusters.

	Paramters:
		args: Namespace
			- contains the arguments from the user
		A:
			- the matrix to factorize
		features:
			- discovered features of the input matrix
	Returns:
		None
	'''
	# Run a desired algorithm to perform non-negative matrix factorization on A
	if args.method == 'all':
		# Initialize W and H
		# Use the same matrices for all the algorithms for comparison
		init_W = np.random.rand(np.size(A, 0), args.num_clusters)
		init_H = np.random.rand(args.num_clusters, np.size(A, 1))

		# Run multiplicative updates with init_W and init_H
		W, H = mu(A, args.num_clusters, delta = 0.0000001, num_iter = args.num_iters, init_W = init_W, init_H = init_H, print_enabled = args.print_enabled)
		print_clusters(W, features, args.cluster_size)

		# Run alternating least squares with init_W and init_H
		W, H = als(A, num_clusters, num_iter = args.num_iters, method = 'als', init_W = init_W, init_H = init_H, print_enabled = args.print_enabled)
		print_clusters(W, features, args.cluster_size)

		# Run alternating non-negative least squares with active set with init_W and init_H
		W, H = als(A, args.num_clusters, num_iter = args.num_iters, method = 'anls_as', init_W = init_W, init_H = init_H, print_enabled = args.print_enabled)
		print_clusters(W, features, args.cluster_size)

	elif args.method == 'mu':
		W, H = mu(A, args.num_clusters, delta = 0.0000001, num_iter = args.num_iters, print_enabled = args.print_enabled)

	elif args.method == 'als':
		W, H = als(A, args.num_clusters, num_iter = args.num_iters, method = 'als', print_enabled = args.print_enabled)

	elif args.method == 'anls':
		W, H = als(A, args.num_clusters, num_iter = args.num_iters, method = 'anls_as', print_enabled = args.print_enabled)

	elif args.method == 'sklearn':
		model = NMF(n_components = args.num_clusters, init='nndsvd')
		W = model.fit_transform(A)
		H = model.components_

	# Print clusters found by non-negative matrix factorization
	if args.method != 'all':
		print_clusters(W, features, args.cluster_size)


if __name__ == '__main__':
	# Process arguments
	args = process_arg()

	# Preprocess the dataset
	try: 
		A, features = preprocess(args.filename, args.col_name, data_frac = args.data_frac, random_sample = args.random_sample, num_max_feature = args.num_max_feature, print_enabled = args.print_enabled)
	except FileNotFoundError:
		sys.exit("Error: File/Column not found")

	process_NMF(args, A, features)
