#!/usr/bin/python
from functools import reduce

from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from mandroid.dataset_preprocessing import load_dataset
from mandroid.dataset_preprocessing import vectorize
from sklearn.metrics import confusion_matrix

"""
    This module is able to check is a given application is an Android malware.
"""


def train_and_validate(X, y):
    """
    Train the model with X, y from dataset and evaluate performance with
    10-fold cross validation. Print vali
    :param X:
    :param y:
    :return:
    """
    clf = LinearSVC(random_state=0)
    X = vectorize(X)
    train_test_SVM(X, y, clf)

    return


def train_test_SVM(X, Y, clf):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    # model = clf.fit(X_train, Y_train)
    # prediction = model.predict(X_test)

    # cross validation
    ris_cross_val = cross_val_score(clf, X, Y, cv=10)

    # confusion matrix
    model = clf.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    conf_mat = confusion_matrix(prediction, Y_test, [0, 1])
    report(ris_cross_val, conf_mat, len(prediction))


def report(cross_val, conf_mat, n_sample):
    r="-----REPORT-----"
    r+= "\nNumber of android apps : " + str(n_sample)
    r += "\n10-Cross validation results :" + str(cross_val)
    r +="\nConfusion matrix:\n"
    r +="true negative : "+ str(conf_mat[0][0])+ "\nfalse negative : "+  str(conf_mat[0][1])+ "\nfalse positive : "+str(conf_mat[1][0])+ "\ntrue positive : "+ str(conf_mat[1][1])

    print r


X, Y = load_dataset("/Users/angelo/Desktop/drebin/feature_vectors",
                   "/Users/angelo/Desktop/drebin/sha256_family.csv", 1000, 45)



train_and_validate(X, Y)
