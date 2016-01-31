#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn import tree

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
#fake_data = preprocess()


def run_dt(data, *arg, **kwarg):
    X_train, X_test, y_train, y_test = data
    clf = tree.DecisionTreeClassifier(*arg, **kwarg)
    s = time()
    clf.fit(X_train, y_train)
    train_time = time() - s
    
    s = time()
    preds = clf.predict(X_test)
    test_time = time() -s
    
    acc = accuracy_score(y_test, preds)
    
    print "\nDecisiton Tree"
    print repr(clf)

    print "train time: ", round(train_time,3)
    print "test time: ", round(test_time,3)
    print "acc: ", round(acc,3)
    
    print "number of features: ", clf.n_features_
    print "tf? :", clf.n_features_ == len(X_train[0])
    
    return acc
    
#run_dt(fake_data, criterion="entropy", min_samples_split=40)

#draw an accuracy graph
accs = []
for n in range(1,50,5):
    
    acc = run_dt(fake_data, criterion="entropy", min_samples_split=n )
    accs.append(acc)

plt.figure()
plt.plot(range(len(accs)), accs)
plt.show()

#########################################################
### your code goes here ###


#########################################################


