import numpy as np

def mu(A, k, num_iter):
	'''
	Run multiplicative updates to perform nonnegative matrix factorization on A.
	Return matrices W, H such that A = WH.
	
	Parameters:
		A: ndarray
			- m by n matrix to factorize
		k: int
			- integer specifying the column length of W / the row length of H
			- the resulting matrices W, H will have sizes of m by k and k by n, respectively
		num_iter: int
			- number of iterations for the multiplicative updates algorithm

	Returns:
		W: ndarray
			- m by k matrix where k = dim
		H: ndarray
			- k by n matrix where k = dim
	'''
	W = np.zeros(np.size(A, 0), k)
	H = np.zeros(k, np.size(A, 1))

	for n in range(num_iter):
		for i in range(np.size())
			


	return W, H