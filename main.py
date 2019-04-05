from preprocessing import preprocess

if __name__ == '__main__':
	A, features = preprocess('abcnews-date-text.csv', 'headline_text', 50)