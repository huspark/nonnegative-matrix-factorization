import numpy as np


def mu(A, k, delta, num_iter):
	'''
	Run multiplicative updates to perform nonnegative matrix factorization on A.
	Return matrices W, H such that A = WH.
	
	Parameters:
		A: ndarray
			- m by n matrix to factorize
		k: int
			- integer specifying the column length of W / the row length of H
			- the resulting matrices W, H will have sizes of m by k and k by n, respectively
		delta: float
			- float that will be added to the numerators of the update rules
			- necessary to avoid division by zero problems
		num_iter: int
			- number of iterations for the multiplicative updates algorithm

	Returns:
		W: ndarray
			- m by k matrix where k = dim
		H: ndarray
			- k by n matrix where k = dim
	'''

	print('Applying multiplicative updates on the input matrix...')
	print('---------------------------------------------------------------------')
	print('Frobenius norm ||A - WH||_F')
	print('')

	V = A
	W = np.random.rand(np.size(A, 0), k)
	H = np.random.rand(k, np.size(A, 1))

	for n in range(num_iter):

		# Update H
		W_TA = W.T @ A
		W_TWH = W.T @ W @ H + delta

		for i in range(np.size(H, 0)):
			for j in range(np.size(H, 1)):
				H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j]

		# Update W
		AH_T = A @ H.T
		WHH_T =  W @ H @ H.T + delta

		for i in range(np.size(W, 0)):
			for j in range(np.size(W, 1)):
				W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j]

		frob_norm = np.linalg.norm(A - W @ H, 'fro')
		print("iteration " + str(n + 1) + ": " + str(frob_norm))

	return W, H


if __name__ == '__main__':
	A = np.matrix([[1, 2, 3], [3, 4, 5]])
	W, H = mu(A, 3, 0.001, 100)