# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 21:01:42 2018

@author: KRISHNA
"""

import pandas as pd
import numpy as np

from sklearn import neighbors, datasets
import matplotlib.pyplot as pt
import numpy as np
from sklearn import svm

'''
data = pd.read_csv('mnist_train.csv')

X = data[:8000]
Y = data[8000:]

np.save('train.npy', X)
np.save('test.npy', Y)
'''
train = np.load('x.npy')
test = np.load('y.npy')

X = train
y = test

'''
X = digits.data
y = digits.target
'''
X_train =X[0:50000]
X_test = X[50000:]
y_train = y[0:50000]
y_test = y[50000:]

lr_model = svm.SVC(kernel='poly')
lr_model.fit(X_train,y_train)
pred = lr_model.predict(X_test)

correct = 0
for i in range(0,len(pred)):
    if pred[i] == y_test[i]:
        correct += 1
print(correct)
print 'accuracy is : ',(float(correct) / len(y_test))*100, '%'
