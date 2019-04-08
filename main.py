#!/usr/bin/env python3

from preprocessing import preprocess
from mu import mu
from als import *
import numpy as np

# def printClusters(A, k):
# 	for i in range(np.size(W, 1)):
# 		cluster = []
# 		idx = (-W[:,i]).argsort()[:5]
# 		for j in range(np.size(idx)):
# 			cluster.append(features[idx[j]])
# 		print(cluster)

if __name__ == '__main__':
	A, features = preprocess('abcnews-date-text.csv', 'headline_text', 100)
	W, H = als(A, 10, 100)
	for i in range(np.size(W, 1)):
		cluster = []
		idx = (-W[:,i]).argsort()[:10]
		for j in range(np.size(idx)):
			cluster.append(features[idx[j]])
		print(cluster)

	print('A is ' + str(np.size(A, 0)) + ' by ' + str(np.size(A, 1)))
	print('W is ' + str(np.size(W, 0)) + ' by ' + str(np.size(W, 1)))
	print('H is ' + str(np.size(H, 0)) + ' by ' + str(np.size(H, 1)))
	print(features)