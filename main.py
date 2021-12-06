import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
# from urllib import request

local_file = 'data.csv'
# url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
# request.urlretrieve(url, local_file)
df = pd.read_csv(local_file)

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)

y = df['custcat'].values

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

k = 4
for i in range(7):
    #Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    yhat[0:5]
    print("Train set Accuracy [k = " + str(k) + "]: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
    print("Test set Accuracy: [k = " + str(k) + "]: ", metrics.accuracy_score(y_test, yhat))
    k += 1
