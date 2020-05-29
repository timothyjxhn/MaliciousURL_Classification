from os.path import isfile

from joblib import load

import utility as ut

model_filename = 'malicious_url_model.pickle'
csv_filename = 'complete_dataset.csv'

def main():
	if not isfile(model_filename):
		print('Waiting for model to load...')
		ut.train_model(csv_filename, model_filename)
	clf = load(model_filename)
	link = ut.tokenize_link(input('Enter link to see if it is malicious: '))
	prediction = clf.predict(link)
	print(f'This URL is {prediction[0]}')


if __name__ == '__main__':
	main()
