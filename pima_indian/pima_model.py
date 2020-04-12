#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:00:06 2020

@author: srajputa010588
"""

#import libs 
from keras.models import Sequential
from keras.layers import Dense
import numpy


#fix seed value 
seedConstant = 7
numpy.random.seed(seedConstant)


# Load poma indian data set
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

#split into x (input) and y (output) dataset 
x = dataset[:,0:8]
y = dataset[:,8]
 
#creating a model 
model = Sequential()

#12 sample record 
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
#8 are sample recored 
model.add(Dense(8,  kernel_initializer='uniform', activation='relu'))
# 1 for output variable 
model.add(Dense(1,  kernel_initializer='uniform', activation='sigmoid'))  


#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#fit the model
model.fit(x, y, epochs=150, batch_size=10)

#Evaluate the model
scores = model.evaluate(x, y)

print ("%s: %.2f%%" %  (model.metrics_names[1], scores[1]*100)) 

