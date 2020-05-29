import csv
import os.path
import re
import string
from urllib.parse import urlparse

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier


def weird_divide(x, y):
	return x / y if y else 0


def get_top_1m():
	top_1m_private = {}
	path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'top-1m.csv')
	top_1m_csv = open(path)
	top_1m_reader = csv.reader(top_1m_csv)
	for row in top_1m_reader:
		top_1m_private[row[1]] = int(row[0])
	return top_1m_private


def get_alexa_rating(domain: str) -> int:
	if re.search(regex, domain):
		return len(top_1m) + 1
	pos = len(top_1m) + 1
	token = domain.split('.')
	for i in range((len(token) * -1) - 1, -1):
		dat = top_1m.get('.'.join(token[i:]))
		if dat is not None:
			pos = dat
			break
	return pos


def get_ccr(netloc: str) -> float:
	longest_alpha = 0
	longest_digit = 0
	longest_symbol = 0
	i = 0
	while i < len(netloc):
		j = i + 1
		if netloc[i].isalpha():
			while j < len(netloc) and netloc[j].isalpha():
				j += 1
			longest_alpha = j - i if j - i > longest_alpha else longest_alpha
		elif netloc[i].isdigit():
			while j < len(netloc) and netloc[j].isdigit():
				j += 1
			longest_digit = j - i if j - i > longest_digit else longest_digit
		else:
			while j < len(netloc) and netloc[j] in string.punctuation:
				j += 1
			longest_symbol = j - i if j - i > longest_symbol else longest_symbol
		i = j
	return (longest_alpha + longest_digit + longest_symbol)/len(netloc)


def train_model(csv_filename, model_filename):
	df = pd.read_csv(csv_filename)
	df.dropna(inplace=True)
	x = df.drop(columns=['url', 'label'])
	y = df['label']
	clf = RandomForestClassifier(max_depth=13, n_jobs=-1, n_estimators=250)
	clf.fit(x, y)
	dump(clf, model_filename)


def tokenize_link(url_string, for_prediction=True, label=None):
	if url_string[-1] != '/':
		url_string += '/'
	url = urlparse(url_string)
	if url.scheme == '':
		url_string = 'http://{}'.format(url_string)
		url = urlparse(url_string)
	url_len = len(url_string.replace('http://', '', 1).replace('https://', '', 1))
	alexa_rating = get_alexa_rating(url.netloc)
	domain_token_count = len(url.netloc.split('.'))
	symbol_domain_count = len([c for c in url.netloc if c in string.punctuation])
	query_digit_count = len([c for c in url.query if c.isdigit()])
	arg_path_ratio = weird_divide(len(url.query), len(url.path))
	arg_domain_ratio = weird_divide(len(url.query), len(url.netloc))
	domain_url_ratio = weird_divide(len(url.netloc), len(url_string))
	path_url_ratio = weird_divide(len(url.path), len(url_string))
	path_domain_ratio = weird_divide(len(url.path), len(url.netloc))
	contains_file = 1 if [i for i in filetypes if i in url.path] else 0
	number_rate_url = len([c for c in url_string if c.isdigit()])
	arg_url_ratio = weird_divide(len(url.query), len(url_string))
	character_continuity_rate = get_ccr(url.netloc)
	contains_redirect = 1 if '//' in url_string.replace('http://', '', 1).replace('https://', '', 1) else 0
	result = {
		'url': url_string,
		'url_len': url_len,
		'alexa_rating': alexa_rating,
		'domain_token_count': domain_token_count,
		'symbol_domain_count': symbol_domain_count,
		'query_digit_count': query_digit_count,
		'domain_url_ratio': domain_url_ratio,
		'path_url_ratio': path_url_ratio,
		'path_domain_ratio': path_domain_ratio,
		'contains_file': contains_file,
		'number_rate_url': number_rate_url,
		'character_continuity_rate': character_continuity_rate,
		'arg_path_ratio': arg_path_ratio,
		'arg_domain_ratio': arg_domain_ratio,
		'arg_url_ratio': arg_url_ratio,
		'contains_redirect': contains_redirect,
		'label': label
	}
	return pd.DataFrame(result, index=[0]).drop(columns=['url', 'label']) if for_prediction else result


top_1m = get_top_1m()
regex = '^(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( ' \
		'25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( ' \
		'25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( ' \
		'25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)'
filetypes = ('.exe', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar', '.ace', '.gz', '.js', '.pdf')