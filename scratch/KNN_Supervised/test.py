from sklearn.model_selection import train_test_split
import pandas as pd

iris_data = pd.read_csv('iris.csv')

X = iris_data.iloc[:,:-1].values
y = iris_data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

from knn import KNN
clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
