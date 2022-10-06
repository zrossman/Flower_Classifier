from kaggle.api.kaggle_api_extended import KaggleApi
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

api = KaggleApi()
api.authenticate()
api.dataset_download_file('arshid/iris-flower-dataset', file_name = 'IRIS.csv')

#Reading in our dataset
df = pd.read_csv('IRIS.csv')

#Lets take a look at our column names and the first row of our dataset
print()
print(df.columns)
print()
print(df.iloc[0, :])
print()
#We can see we have four potential inputs(features) and one label, the species type.

#Let's turn our features and our label into arrays
X = np.array(df.iloc[:, :4])
y = np.array(df.iloc[:, 4])

#Now we split out data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

#Now, since this algorithm computes distance and also assumes normality, we need to scales our features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Now lets define our classifier model: K-Nearest Neighbors. First, we find k.
k = math.sqrt(len(df))
print(k)
#We find k to be 12.24. Since our model is going to use K-Nearest Neighbors, we need k to be odd. So we'll round to
#the nearest odd integer.
k = math.ceil(k)
print(k)
print()

#Now, we can define our model. We have four classes, so p = 4.
classifier = KNeighborsClassifier(n_neighbors = 13, p = 4, metric = 'euclidean')
classifier.fit(X_train, y_train)

#Predicting the test results
y_pred = classifier.predict(X_test)
print(y_pred[:10])
print()

#Evaluating our model
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f1_score(y_test, y_pred, average = 'micro'))
print(accuracy_score(y_test, y_pred))

#Comparing our y_pred to our y_test
comparison = []
for i in range(len(y_test)):
    a_list = []
    a_list.append(y_pred[i])
    a_list.append(y_test[i])
    comparison.append(a_list)
print('Comparison (prediction, actual):', comparison)
print()

#Using the k-Nearest Neighbors algorithm, we are able to regularly predict with accuracy in the 90's. 


