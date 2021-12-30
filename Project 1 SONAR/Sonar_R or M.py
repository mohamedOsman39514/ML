# Importing Dependences
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and data Processing
sonarData = pd.read_csv('sonar_data.csv', header=None)

# Number of Rows and Columns
sonarData.shape

# we describe the all of data that calculate the: mean , standerd deviation, min , max , count --> statistical
sonarData.describe()

# Sperating data and labels
X = sonarData.drop(columns=60,axis=1)
y = sonarData[60]

# traninig and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)

# Model Traning
model = LogisticRegression()
model.fit(x_train, y_train)

# accurecy of traning data
x_train_pre = model.predict(x_train)
traninig_data_acc = accuracy_score(x_train_pre, y_train)
print(traninig_data_acc)

# accurecy of testing data
x_test_pre = model.predict(x_test)
testing_data_acc = accuracy_score(x_test_pre, y_test)
print(testing_data_acc)