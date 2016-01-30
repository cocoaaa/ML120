# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:50:30 2016

@author: hjsong
"""

#visualization
import random
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

def prettyPicture(trainedClf, X_test, y_test):
    #draw the mesh and decision boundary
    x_min = 0, x_max = 1, y_min = 0, y_max =1
    xx,yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    coordinates = [point in zip(xx_ravel, yy_ravel)]
    Z = clf.predict(coordinates)
    Z = Z.reshape(xx.shape)
    
    plt.figure(0)
    plt.psoclormesh(xx,yy,Z,cmap=pl.cm.bwr)
    plt.show(2)
    