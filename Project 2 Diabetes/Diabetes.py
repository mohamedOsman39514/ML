from _ast import Is
from itertools import starmap

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

# read data
dataset = pd.read_csv('diabetes.csv')
x = dataset.iloc[:, :8]
y = dataset.iloc[:, 8]
# scaling the data
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
#y = sc_y.fit_transform(y)
y = np.reshape(a=y , newshape=(-1,1))
y = np.real(sc_y.fit_transform(y))


# split data to train and test
x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
#traning model
classifier = svm.SVR(kernel='linear')
classifier.fit(x_train, y_train)

