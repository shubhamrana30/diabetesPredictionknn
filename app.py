# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 01:37:24 2020

@author: Asus
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    '''text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
    text8 = request.form['8']'''
    x=[np.array(int_features)]
    x_train=pd.read_csv('file:///‪C:/Users/Asus/Desktop/trainDiabetes.csv')
    x_train=x_train.iloc[:,1:9]
    #print(x_train)
    standard=StandardScaler()
    x_train=standard.fit_transform(x_train)
    x=standard.transform(x)
    y_predict=model.predict(x)
    #row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8])])
    prediction=y_predict[0]
    print(y_predict)
    #output='{0:.{1}f}'.format(prediction[0][1], 2)

    if prediction==0:
        return render_template('index.html',pred='You are safe.\n Probability of having diabetes is very low')
    else:
        return render_template('index.html',pred='You have a high chances of having diabetes')
if __name__ == "__main__":
    app.run(port=3000,debug=True)

