# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:55:26 2016

@author: hjsong
"""
import random
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import mytools


#make fake dataset
def make_fake_data():
    random.seed(42)
    n_points = 100;
    bumpiness = [random.random() for i in xrange(0,n_points)]
    grade = [random.random() for i in xrange(0,n_points)]
    
    y = [round(bumpiness[i]*grade[i]+0.3+0.1) for i in xrange(0,n_points)]
    for i in xrange(0,n_points):
        if grade[i] > 0.8 and bumpiness[i]>0.8:
            y[i] = 1.0
    #print y
    #split to test and train
    X = [[gg, bb] for gg,bb in zip(grade, bumpiness)]
    split = int(0.75*n_points)
    X_train = X[0:split]
    X_test = X[split:]
    y_train = y[0:split]
    y_test = y[split:]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = mytools.make_fake_data()
#scatter plot
xx=[]
yy=[]
colors=[]
for idx, point in enumerate(X):
    xx.append(point[0])
    yy.append(point[1])
    if y[idx] == 1:
        colors.append('blue') #fase
    else:
        colors.append("yellow")
plt.figure(0)
plt.title("Training data")
plt.xlabel('grade')
plt.ylabel('bumpiness')
plt.xlim(0,1)
plt.ylim(0,1)
plt.scatter(xx, yy, c=colors)
plt.hold(True)
plt.show()


##Initialize GaussianNB classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

##Test and calculate %accuracy
#preds = clf.predict(X_test)
#n_test = len(X_test)
#accs = [preds[i] == y_test[i] for i in range(0,n_test)]
#print sum(accs) / float(len(accs))
#
##Alternatively, use built-in function
#print clf.score(X_test, y_test)

#Plot predictions: black if correctly labeled, red otherwise
#plt.figure(0)
#plt.xlabel("grade"); plt.ylabel("bumpiness")
#for i, point in enumerate(X_test):
#    color = "black"
#    if not accs[i]: #wrong classification
#        color = "red"
#    plt.scatter(point[0], point[1], marker = u'x', c=color)
#plt.show()

#viz


mytools.prettyPicture(clf, X_test, y_test)
