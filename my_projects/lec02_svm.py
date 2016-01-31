# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 17:21:43 2016

@author: hjsong
"""

import random, time
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import mytools
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def draw_svm_db(kernel="linear", gamma=1000, C=1.0):
    
    #Train
    X_train, X_test, y_train, y_test = mytools.make_fake_data()
    clf = SVC(kernel=kernel, gamma=gamma, C=C)
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    
    #Predict
    start = time.time()
    preds = clf.predict(X_test)
    test_time = time.time() - start 
    
    #Score
    accuracy = accuracy_score(y_test, preds)
    
    print ""
    print "Kernel: ", kernel
    print "C: ", C, " , gamma: ", gamma
    print "Accuracy: ", accuracy
    print "Training time: ", round(train_time,3)
    print "Test time: ", round(test_time,3)
    
    #Visualization
    mytools.prettyPicture(clf, X_test, y_test)
    plt.show()

#Different kernels
#for k in ["linear", "poly", "rbf", "sigmoid" ]:
#    draw_svm_db(k)

#Different gammas for rbf
#gamma does nothing in linear kernel
#for gamma in range(0,1000,50):
#    draw_svm_db(kernel="rbf", gamma=gamma)
    
for C in [0.1,100,1000]:
    if C>0:
        draw_svm_db(kernel="rbf",C=C)

