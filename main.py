#!/usr/bin/env python3

from preprocessing import preprocess
from mu import mu
from als import *
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer

# def printClusters(A, k):
# 	for i in range(np.size(W, 1)):
# 		cluster = []
# 		idx = (-W[:,i]).argsort()[:5]
# 		for j in range(np.size(idx)):
# 			cluster.append(features[idx[j]])
# 		print(cluster)

# def get_nmf_topics(model, features, n_top_words):
    
#     #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
#     #feat_names = vectorizer.get_feature_names()
    
#     word_dict = {};
#     for i in range(10):
        
#         #for each topic, obtain the largest values, and add the words they map to into the dictionary.
#         words_ids = model.components_[i].argsort()[:-20 - 1:-1]
#         words = [features[key] for key in words_ids]
#         word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
#     return pd.DataFrame(word_dict);

if __name__ == '__main__':

	A, features = preprocess(filename = 'abcnews-date-text.csv', col_name = 'headline_text', data_frac = 0.01, num_max_feature = 1000)

	model = NMF(n_components=10, init='nndsvd')
	W = model.fit_transform(A)
	H = model.components_

	#W, H = als(A, 10, 1000)
	for i in range(np.size(W, 1)):
		cluster = []
		idx = (-W[:,i]).argsort()[:10]
		for j in range(np.size(idx)):
			cluster.append(features[idx[j]])
		print(cluster)

	print('A is ' + str(np.size(A, 0)) + ' by ' + str(np.size(A, 1)))
	print('W is ' + str(np.size(W, 0)) + ' by ' + str(np.size(W, 1)))
	print('H is ' + str(np.size(H, 0)) + ' by ' + str(np.size(H, 1)))
	#print(features)