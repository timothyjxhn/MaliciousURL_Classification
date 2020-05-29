from statistics import mean

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('complete_dataset.csv')
df.dropna(inplace=True)

x = df.drop(columns=['url', 'label'])
y = df['label']

kf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
accuracy_arr = []

for train_index, test_index in kf.split(x, y):
	clf = RandomForestClassifier(n_estimators=175, max_depth=13, n_jobs=-1)
	x_train, x_test = x.iloc[train_index], x.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	clf.fit(x_train, y_train)

	y_pred = clf.predict(x_test)
	accuracy_arr.append(accuracy_score(y_test, y_pred))

	conf_mat = plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues, normalize='true')
	conf_mat.ax_.set_title(f'Confusion Matrix with {len(y_test)} samples')
	plt.show()

print('Mean Accuracy: {}%'.format(mean(accuracy_arr) * 100))
