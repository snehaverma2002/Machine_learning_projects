import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('iris.csv')

X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:,-1])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)


pickle.dump(knn, open('iris.pkl', 'wb'))

model = pickle.load(open('iris.pkl','rb'))
