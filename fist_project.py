import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('vijay.csv')

v = df.iloc[: , 2:34]

new_df = v.dropna(thresh=25)

p =  new_df.drop(columns=['Recording name', 'Recording date' , 'Recording Fixation filter name'])

vijay_1 = p.loc[p['Participant_name'] == "VIJAY1"]
vijay_2 = p.loc[p['Participant_name'] == "VIJAY2"]


l = vijay_1.Recording_timestamp
 
my_list = l.tolist()

my_list = [my_list[i+1] - my_list[i] for i in range(len(my_list) -1)]

y = pd.DataFrame(my_list)

vijay_1 = vijay_1.drop(columns=['Participant_name','Recording duration','Recording_timestamp'])

vijay_1 = vijay_1.drop(columns=['Recording start time'])

x = vijay_1

x = x[:-1]

#start

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(x_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


