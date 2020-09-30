# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:48:17 2020

@author: Asus
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


url='file:///‪C:/Users/Asus/Desktop/diabetes.csv'
dataset=pd.read_csv(url)

#changing data set values

#dataset["Pregnancies"]=dataset["Pregnancies"].astype("int")
zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0,np.nan)
    mean = int(dataset[column].mean(skipna = True))
    dataset[column] = dataset[column].replace(np.nan,mean)
 
#dataset.to_json("C:/Users/Asus/Desktop/helloshub.json")


x = dataset.iloc[:,:8]
y = dataset.iloc[:,8]


x_train, x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size = 0.2)

mean=x_train.mean()
std=np.std(x_train)
print(std)
print(mean)
standard = StandardScaler()
#x_train.to_csv("C:/Users/Asus/Desktop/trainDiabetes.csv")
x_train = standard.fit_transform(x_train)
x_test=[1,89,66,23,94,28.1,0.167,21]
#[0,137,40,35,168,43.1,2.288,33]
#[1,89,66,23,94,28.1,0.167,21]
x_test=[np.array(x_test)]
x_test = standard.transform(x_test)
#print(x_test)

knn = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')

knn.fit(x_train,y_train)

#pickle.dump(knn,open('model.pkl','wb'))

y_predict = knn.predict(x_test)
#print(y_predict)

#cm = confusion_matrix(y_test,y_predict)
#print(cm)
#print(accuracy_score(y_test,y_predict))
