import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# import dataset and show first few columnns
df = pd.read_csv('abcnews-date-text.csv')
print('\n')
print(df.head())
print('\n')
#print(df['headline_text'].head())


# trasform the dataset into a non-negative matrix A
head_temp = df['headline_text'].head()
vectorizer = CountVectorizer()
A = vectorizer.fit_transform(head_temp).toarray()

print(A)
print('\n')
print(vectorizer.get_feature_names())
print('\n')

# trasform the dataset into a non-negative matrix A
headlines = df['headline_text']
vectorizer = CountVectorizer()
A = vectorizer.fit_transform(headlines).toarray()

print(A[1:10, 1:10])
print('\n')
#print(vectorizer.get_feature_names())
#print('\n')