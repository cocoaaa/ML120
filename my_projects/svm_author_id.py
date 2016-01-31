#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mytools
sys.path.append("../ud120-projects-start/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
kernel = "linear"
clf = SVC(kernel=kernel)
start = time()
clf.fit(features_train, features_test)
train_time = time() - start

start = time()
preds = clf.predict(features_test)
test_time = time() - start

acc = accuracy_score(labels_test, preds)

print "train time: ", train_time
print "test time: ", test_time
print "accuracy: ", acc

mytools.prettyPicture(clf, features_test, labels_test)





#########################################################
### your code goes here ###

#########################################################


