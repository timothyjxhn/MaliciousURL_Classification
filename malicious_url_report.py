import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


df = pd.read_csv('complete_dataset.csv')
df.dropna(inplace=True)

x = df.drop(columns=['url', 'label'])
y = df['label']
labels = ['benign', 'malware', 'phishing']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.1)

clf = RandomForestClassifier(n_estimators=175, max_depth=13, n_jobs=-1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred, target_names=labels))
