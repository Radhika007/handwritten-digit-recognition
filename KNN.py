#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 18:30:46 2019

@author: radhika
"""

from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("Loading data")
mndata = MNIST()
images, labels = mndata.load_training()

clf = KNeighborsClassifier()

#Training the dataset
X_train = images[:10000]
y_train = labels[:10000]

#Fitting the data into the model
clf.fit(X_train, y_train)

#Testing our model on the next 100 images
X_test = images[10000:10100]
expected = labels[10000:10100].tolist()

#Prediction
predicted = clf.predict(X_test)

#Accuracy rate
print("Accuracy: ", accuracy_score(expected,predicted))