import numpy as np
from numpy.linalg import lstsq


def as_vector(B, y, eps = 0.0001):
	'''
	Use the active set method to solve a least squares problem ||Bg - y||_2.
	Here, B is a matrix and y is a vector.
	
	Parameters:
		B: ndarray
			- m by n matrix
		y: ndarray
			- m by 1 vector
		eps: float
			- constant

	Returns:
		g: ndarray
			- n by 1 vector solution to the least squares problem ||Bg - y||_2
	'''

	# Initialize variables
	# E is a variable that keeps track of indices for the active set
	# S is a variable that keeps track of indices for the passive set
	n = np.size(B, 1)
	g = np.zeros(n)
	E = np.ones(n)
	S = np.zeros(n)
	w = (B.T).dot(y - B.dot(g))

	wE = w * E
	t = np.argmax(wE)
	v = wE[t]

	while np.sum(E) > 0 and v > eps:
		# Update active/passive set indices
		E[t] = 0
		S[t] = 1

		Bs = B[:, S > 0]
		zsol = lstsq(Bs, y, rcond = -1)[0]
		zz = np.zeros(n)
		zz[S > 0] = zsol
		z = zz + 0

		while np.min(z[S > 0]) <= 0:
			alpha = np.min((g / (g - z))[(S > 0) * (z <= 0)])
			g += alpha * (z - g)
			S[g == 0] = 0
			E[g == 0] = 1
			Bs = B[:, S > 0]
			zsol = lstsq(Bs, y)[0]
			zz = np.zeros(n)
			zz[S > 0] = zsol
			z = zz

		g = z
		w = (B.T).dot(y - B.dot(g))
		wE = w * E
		t = np.argmax(wE)
		v = wE[t]
	return g


def as_matrix(B, Y):
	'''
	Use the active set method to solve a least squares problem ||BG - Y||_2.
	Here, B and Y are matrices.
	
	Parameters:
		B: ndarray
			- m by n matrix
		Y: ndarray
			- n by l matrix

	Returns:
		G: ndarray
			- n by l matrix solution to the least squares problem ||BG - Y||_2
	'''

	# Use the active set method (as_vector) to solve least square problems on the matrix B and each column of Y
	return [np.array([as_vector(B, column) for column in Y.T]).T]


def anls_as(A, k, num_iter, init_W = None, init_H = None, print_enabled = False):
	'''
	Run the alternating nonnegative least squares algortihm with the active set method 
	to perform nonnegative matrix factorization on A.
	Return matrices W, H such that A = WH.

	Parameters:
		A: ndarray
			- m by n matrix to factorize
		k: int
			- integer specifying the column length of W / the row length of H
			- the resulting matrices W, H will have sizes of m by k and k by n, respectively
		num_iter: int
			- number of iterations for the multiplicative updates algorithm
		print_enabled: boolean
			- if ture, output print statements

	Returns:
		W: ndarray
			- m by k matrix where k = dim
		H: ndarray
			- k by n matrix where k = dim
	'''

	print('Applying the alternating non-negative least squares algorithm with the active set method on the input matrix...')

	if print_enabled:
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

	# Decompose the input matrix
	for n in range(num_iter):
		H = as_matrix(W, A)[0]
		W = as_matrix(H.T, A.T)[0].T

		if print_enabled:
			frob_norm = np.linalg.norm(A - W @ H, 'fro')
			print("iteration " + str(n + 1) + ": " + str(frob_norm))

	return W, H