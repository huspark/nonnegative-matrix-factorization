import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
import glob

def preprocess_bbcsport(num_max_feature, print_enabled = True):
	''' 
	Preprocess a column of a csv file for nonnegative matrix factorization.

	Parameters:
		max_feature_num: int
			- integer specifing the maximum number of extracted features
		print_enabled: boolean
			- if ture, output print statements

	Returns:
		A: ndarray
			- (# of features) by (# of datapoints) ndarray that will be used for NMF
		features: list
			- list with length of (# of features) containing strings of extracted features
	'''

	# Import dataset
	files = []
	data = []

	for f in glob.glob('bbcsport/athletics/*.txt'):
		files.append(f)
	for f in glob.glob('bbcsport/cricket/*.txt'):
		files.append(f)
	for f in glob.glob('bbcsport/football/*.txt'):
		files.append(f)
	for f in glob.glob('bbcsport/rugby/*.txt'):
		files.append(f)
	for f in glob.glob('bbcsport/tennis/*.txt'):
		files.append(f)

	for filename in files:
		with open(filename, 'rb') as file:
			text = file.read().decode(errors='ignore').replace("\n\n", ' ').replace("\n", ' ')
			data.append([ filename, text ])

	# create data frame
	df = pd.DataFrame(data, columns = ['file' , 'article'])

	# Transform the specified column of the given csv file into a matrix A.
	# Store the discovered features to features.
	articles = df['article']
	vectorizer = CountVectorizer(max_features = num_max_feature, stop_words = 'english')
	count_mat = vectorizer.fit_transform(articles)
	transformer = TfidfTransformer(smooth_idf = False)

	A = transformer.fit_transform(count_mat).toarray().T
	features = vectorizer.get_feature_names()

	if print_enabled:
		# Print out results.
		print('---------------------------------------------------------------------')
		print(str(np.size(A, 0)) + ' by ' + str(np.size(A, 1)) + ' matrix\n')
		print(A)
		print('')

		print('---------------------------------------------------------------------')
		print(str(len(features)) + ' features\n')
		print(features)
		print('')

	return A, features