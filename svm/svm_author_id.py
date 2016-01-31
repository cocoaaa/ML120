#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#n = len(features_train)/100
#features_train = features_train[:n]
#labels_train = labels_train[:n]
kernel = "rbf"
data = [features_train, features_test, labels_train, labels_test]


def run_svm(data, kernel='linear', C=1.0, gamma = 'auto'):
    features_train, features_test, labels_train, labels_test = data
    clf = SVC(kernel=kernel, C=C, gamma=gamma)
    
    start = time()
    clf.fit(features_train, labels_train)
    train_time = time() - start
    
    start = time()
    preds = clf.predict(features_test)
    test_time = time() - start
    
    acc = accuracy_score(labels_test, preds)
    
    print ""
    print "kernel: ", kernel
    print "C: ", C, ", gamma: ", gamma
    print "train time: ", round(train_time,3)
    print "test time: ", round(test_time,3)
    print "accuracy: ", round(acc,3)
    
    print "10th: ", preds[10]
    print "26th: ", preds[26]
    print "50th: ", preds[50]
    print "sum: ", sum(preds)
    return acc

#mytools.prettyPicture(clf, features_test, labels_test)

#accs = []
#for C in [10,100,1000,10000]:
#    acc = run_svm(data, kernel='linear', C=C, gamma = 1.0)
#    accs.append(acc)
#plt.plot()
#plt.plot(range(len(accs)), accs)
#plt.show()

run_svm(data, 'rbf', C= 10000)
#
#accs = []
#for g in np.arange(1, 1000, 100):
#    acc = run_svm(data, kernel='linear', gamma=g)
#    accs.append(acc)
#plt.plot()
#plt.plot(range(len(accs)), accs)
#plt.show()


#########################################################
### your code goes here ###

#########################################################


