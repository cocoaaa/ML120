# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:50:30 2016

@author: hjsong
"""
import random
#visualization
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

def prettyPicture(clf, X_test, y_test):
    #draw the mesh and decision boundary
    x_min = 0; x_max = 1; y_min = 0; y_max =1;h=0.01;
    xx,yy = np.meshgrid(np.arange(x_min, x_max,h), np.arange(y_min, y_max,h))
    coordinates = [point for point in zip(xx.ravel(), yy.ravel())]
    Z = clf.predict(coordinates)
    Z = Z.reshape(xx.shape)
#    colors = ['green', 'red']
    
    
    #Plot decision boundaries    
    plt.figure()
    plt.title("Decision Boundary, Test result")
    plt.xlabel("feature1")
    plt.ylabel("feature2")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.pcolormesh(xx,yy,Z,cmap=pl.cm.seismic)
    plt.hold(True)
    #Plot the predictions
    #Orange means wrong classification
    pcolors = ['orange', 'black']
    preds = clf.predict(X_test)
    tf = (preds == y_test)
    for i,point in enumerate(X_test):
        plt.scatter(point[0], point[1], marker = u'x', c = pcolors[int(tf[i])])
#    return plt.figure(0)
    plt.show()
    
    
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