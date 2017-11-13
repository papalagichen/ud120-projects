#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
if __name__ == '__main__':
    from sklearn import svm

    # features_train = features_train[:len(features_train) / 100]
    # labels_train = labels_train[:len(labels_train) / 100]

    classifier = svm.SVC(C=10000.0)
    t0 = time()
    classifier.fit(features_train, labels_train)
    print "training time: ", round(time() - t0, 3), "s"

    t1 = time()
    predict = classifier.predict(features_test)
    print reduce(lambda x, y: x + y, predict)
    print "predicting time: ", round(time() - t1, 3), "s"

    t2 = time()
    score = classifier.score(features_test, labels_test)
    print "scoring time: ", round(time() - t2, 3), "s"
    print score
#########################################################



