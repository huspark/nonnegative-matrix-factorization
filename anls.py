#!/usr/bin/env python3

import numpy as np
import numpy.linalg
import scipy.optimize

# import scipy.optimize
# B = np.loadtxt("B.txt", delimiter=',')
# b = np.loadtxt("b.txt", delimiter=',')

def nnls_as_matrix_vector(B, b, eps):
  m,n = np.shape(B)
  # g = np.zeros([n,1])
  g = np.zeros(n)
  S = np.zeros(n)
  E = np.ones (n)
  # print ("shape is ", np.shape(B), np.shape(B.T))
  w = (B.T).dot(b-B.dot(g))

  w_dot_E = w * E
  t = np.argmax(w_dot_E)
  v = w_dot_E[t]
  while np.sum(E)>0 and v > eps:
    E[t]=0
    S[t]=1
    BS = B[:,S>0]
    zsol = np.linalg.lstsq(BS,b, rcond=None)[0]
    zz = np.zeros(n)
    zz[S>0] = zsol
    z = zz + 0
 

    while np.min(z[S>0]) <= 0:
      alpha = np.min((g / (g-z))[(S>0)*(z<=0)])
      g += alpha * (z-g)
      S[g==0] = 0
      E[g==0] = 1
      BS = B[:,S>0]
      zsol = np.linalg.lstsq(BS,b)[0]
      zz = np.zeros(n)
      zz[S>0] = zsol
      z = zz + 0
    g = z + 0
    w = (B.T).dot(b-B.dot(g))
    w_dot_E = w*E
    t = np.argmax(w_dot_E)
    v = w_dot_E[t]
  return g

def nnls_as(B, BB):
  eps = 0.00001
  return [np.array([nnls_as_matrix_vector(B,column, eps) for column in BB.T]).T]


# B = np.random.rand(10,5)
# B2 = np.random.rand(10,2)
# B1 = np.random.rand(10,1)
# RET = nnls_as(B,B2,0.00001)
# B.dot(RET[0])-B2
# np.array([(B.dot(((scipy.optimize.nnls(B,B2[:,0]))[0]))-B2[:,0]), (B.dot(((scipy.optimize.nnls(B,B2[:,1]))[0]))-B2[:,1])]).T
