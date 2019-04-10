import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF



def preprocess(filename, col_name, data_frac, num_max_feature):
	''' 
	Preprocess a column of a csv file for nonnegative matrix factorization.

	Parameters:
		filename: str
		col_name: str
		data_frac: float
			- float in [0, 1], specifying the fraction of data we will be using
		max_feature_num: int
			- integer specifing the maximum number of extracted features

	Returns:
		A: ndarray
			- (# of features) by (# of datapoints) ndarray that will be used for NMF
		features: list
			- list with length of (# of features) containing strings of extracted features
	'''

	# Import dataset and show first few columnns.
	df = pd.read_csv(filename)

	# Experiment with 10000 data points
	# df = df.sample(frac = data_frac, random_state = 1)
	df = df.sample(frac = data_frac)

	# print('Prerpocessing ' + col_name + ' column of ' + filename + ' file...')
	# print('---------------------------------------------------------------------')
	# print('First few lines of ' + filename +'\n')
	# print(df.head())
	# print('')

	# Transform the specified column of the given csv file into a matrix A.
	# Store the discovered features to features.
	headlines = df['headline_text']
	vectorizer = CountVectorizer(max_features = num_max_feature, stop_words = 'english')
	count_mat = vectorizer.fit_transform(headlines)
	transformer = TfidfTransformer(smooth_idf=False)

	A = transformer.fit_transform(count_mat).toarray().T
	features = vectorizer.get_feature_names()

	# # Print out results.
	# print('---------------------------------------------------------------------')
	# print(str(np.size(A, 0)) + ' by ' + str(np.size(A, 1)) + ' matrix representing ' + col_name + ' column of ' + filename + ' file\n')
	# print(A)
	# print('')

	# print('---------------------------------------------------------------------')
	# print(str(len(features)) + ' features of ' + col_name + ' column of ' + filename + ' file\n')
	# print(features)
	# print('')

	return A, features


if __name__ == "__main__":
	preprocess('abcnews-date-text.csv', 'headline_text', 0.01, 1000)