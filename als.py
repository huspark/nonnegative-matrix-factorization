import numpy as np
from numpy.linalg import lstsq


def nls_as_matrix_vector(B, b, eps):
	n = np.size(B,1)

	# Initialize variables
	g = np.zeros(n)
	E = np.ones (n)
	S = np.zeros(n)
	w = (B.T).dot(b - B.dot(g))

	w_dot_E = w * E
	t = np.argmax(w_dot_E)
	v = w_dot_E[t]

	while np.sum(E) > 0 and v > eps:
		E[t] = 0
		S[t] = 1
		BS = B[:, S > 0]
		zsol = np.linalg.lstsq(BS, b, rcond = None)[0]
		zz = np.zeros(n)
		zz[S > 0] = zsol
		z = zz + 0

		while np.min(z[S > 0]) <= 0:
			alpha = np.min((g / (g - z))[(S > 0) * (z <= 0)])
			g += alpha * (z - g)
			S[g == 0] = 0
			E[g == 0] = 1
			BS = B[:, S > 0]
			zsol = np.linalg.lstsq(BS, b)[0]
			zz = np.zeros(n)
			zz[S > 0] = zsol
			z = zz + 0

		g = z + 0
		w = (B.T).dot(b - B.dot(g))
		w_dot_E = w*E
		t = np.argmax(w_dot_E)
		v = w_dot_E[t]
	return g


def nls_as(B, M):
  eps = 0.00001
  return [np.array([nls_as_matrix_vector(B, column, eps) for column in M.T]).T]


def als(A, k, num_iter, method = ['als', 'anls_as'], init_W = None, init_H = None):
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
		method: string
			- string specifying which method to use to solve least square problems

	Returns:
		W: ndarray
			- m by k matrix where k = dim
		H: ndarray
			- k by n matrix where k = dim
	'''

	if method == 'als':
		print('Applying the alternating least squares method on the input matrix...')
	elif method == 'anls_as':
		print('Applying the alternating non-negative least squares with active set method on the input matrix...')

	print('---------------------------------------------------------------------')
	print('Frobenius norm ||A - WH||_F')
	print('')

	# Initialize W and H
	if init_W is None:
		W = np.random.rand(np.size(A, 0), k)
	else:
		W = init_W

	if init_H is None:
		H = np.random.rand(k, np.size(A, 1))
	else:
		H = init_H

	for n in range(num_iter):
		# Use the alternating least squares method to solve LS
		if method == 'als':
			# Update H
			# Solve the least squares problem: argmin_H ||WH - A||
			H = lstsq(W, A, rcond = None)[0]
			# Set negative elements of H to 0
			H[H < 0] = 0

		    # Update W
			# Solve the least squares problem: argmin_W.T ||H.TW.T - A.T||
			W = lstsq(H.T, A.T, rcond = None)[0].T

			# Set negative elements of W to 0
			W[W < 0] = 0

		# Use the alternating non-negative least squares method with an active set to solve LS
		elif method == 'anls_as':
			H = nls_as(W, A)[0]
			W = nls_as(H.T, A.T)[0].T

		frob_norm = np.linalg.norm(A - W @ H, 'fro')
		print("iteration " + str(n + 1) + ": " + str(frob_norm))

	return W, H


if __name__ == '__main__':
	A = np.matrix([[1, 2, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 2, 1]])
	W, H = als(A, 2, 100)