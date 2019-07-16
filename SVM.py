#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:53:41 2019

@author: radhika
"""
from mnist import MNIST
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("Data is loading...")
dataset = MNIST()
images, labels = dataset.load_training()

classifier = SVC(kernel = 'linear', random_state = 0)

#Training the dataset
X_train = images[:10000]
y_train = labels[:10000]

#Fitting the data into the model
classifier.fit(X_train, y_train)

#Testing our model on the next 100 images
X_test = images[10000:10100]
expected = labels[10000:10100].tolist()

#Prediction
predicted = classifier.predict(X_test)

#Accuracy rate
print("Accuracy: ", accuracy_score(expected,predicted))