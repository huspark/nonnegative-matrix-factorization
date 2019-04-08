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

	print('Applying multiplicative updates on the input matrix...')
	print('---------------------------------------------------------------------')

	V = A
	W = np.ones((np.size(A, 0), k))
	H = np.ones((k, np.size(A, 1)))

	for n in range(num_iter):
		# Update W
		VH_T = V @ H.T
		WHH_T =  W @ H @ H.T

		for i in range(np.size(W, 0)):
			for j in range(np.size(W, 1)):
				W[i, j] = W[i, j] * VH_T[i, j] / WHH_T[i, j]

		# Update H
		W_TV = W.T @ V
		W_TWH = W.T @ W @ H

		for i in range(np.size(H, 0)):
			for j in range(np.size(H, 1)):
				H[i, j] = H[i, j] * W_TV[i, j] / W_TWH[i, j]

		print("iteration " + n + "...")

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
	A = np.matrix([[1, 2], [3, 4]])
	W, H = mu(A, 3, 100)