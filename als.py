import numpy as np
from numpy.linalg import lstsq
from anls import nnls_as

def anls(A, k, num_iter):
  return als(A,k,num_iter, nnls_as)

def als(A, k, num_iter, method = 'lstsq'):
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

	print('Applying the alternating least squares method on the input matrix...')
	print('---------------------------------------------------------------------')

	W = np.ones((np.size(A, 0), k))
	H = np.ones((k, np.size(A, 1)))

	for n in range(num_iter):
		if method == 'lstsq':
			# Update H
			# Solve the least squares problem: argmin_H ||WH - A||
			H = lstsq(W, A)[0]
			# Set negative elements of H to 0
			H[H < 0] = 0

		        # Update W
			# Solve the least squares problem: argmin_W.T ||H.TW.T - A.T||
			W = lstsq(H.T, A.T)[0].T

			# Set negative elements of W to 0
			W[W < 0] = 0
		else:
			H = method(W, A)[0]
			W = method(H.T, A.T)[0].T
		print("iteration " + str(n) + "...")

	print('A = ')
	print(A)
	print('W = ')
	print(W)
	print('H = ')
	print(H)
	print('W * H = ')
	print(W @ H)

	return W, H

if __name__ == '__main__':
	A = np.matrix([[1, 2, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 2, 1]])
	W, H = als(A, 2, 100)