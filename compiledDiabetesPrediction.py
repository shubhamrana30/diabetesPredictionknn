# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:59:19 2020

@author: Asus
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

model=pickle.load(open('model.pkl','rb'))

#x_test=[0,137,40,35,168,43.1,2.288,33]
x_test=[1,89,66,23,94,28.1,0.167,21]

'''
for i in range(8):
    val=input('enter')
    x_test.append(int(val))
for i in x_test:
    print(i)
'''
standard=StandardScaler()
x=[np.array(x_test)]

x_train=pd.read_csv('file:///‪C:/Users/Asus/Desktop/trainDiabetes.csv')
x_train=x_train.iloc[:,1:9]
#print(x_train)
x_train=standard.fit_transform(x_train)
x=standard.transform(x)
y_predict=model.predict(x)
print(y_predict)


'''
import nltk
import sklearn

print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
'''