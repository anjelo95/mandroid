#!/usr/bin/python
from sklearn import  linear_model
import numpy as np
import matplotlib.pyplot as plt
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
    #clf = LinearSVC(random_state=0)
    clf= linear_model.SGDClassifier(max_iter=100)
    X = vectorize(X)


    return train_test_SVM(X, y, clf)


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
    return report(ris_cross_val, conf_mat, len(prediction))


def report(cross_val, conf_mat, n_sample):
    r="-----REPORT-----"
    r+= "\nNumber of android apps : " + str(n_sample)
    r += "\n10-Cross validation results :" + str(cross_val)
    r +="\nConfusion matrix:\n"
    r +="true negative : "+ str(conf_mat[0][0])+ "\nfalse negative : "+  str(conf_mat[0][1])+ "\nfalse positive : "+str(conf_mat[1][0])+ "\ntrue positive : "+ str(conf_mat[1][1])

    print r
    return cross_val,conf_mat



def plot_result():
    # plot
    min_range=10000
    max_range =10001
    step=2
    x_value=[]
    y_value=[]
    conf_matrices=[]
    for i in range(min_range,max_range,step):


        X, Y = load_dataset("/Users/angelo/Desktop/drebin/feature_vectors",
                        "/Users/angelo/Desktop/drebin/sha256_family.csv", i, 30)
        cross_val, conf_mat=train_and_validate(X, Y)
        x_value.append(i)
        y_value.append(np.mean(cross_val))
        #conf_matrices.append(conf_mat)
        plot_conf_mat(conf_mat)

    plt.plot(x_value, y_value, 'ro')
    plt.axis([0, max_range+100, 0.8, 1.0],'equal')
    plt.ylabel('average accuracy on k=10 cross validation')
    plt.xlabel('dimension of dataset ')
    plt.show()






'''
X, Y = load_dataset("/Users/angelo/Desktop/drebin/feature_vectors",
                   "/Users/angelo/Desktop/drebin/sha256_family.csv", 1000, 45)



train_and_validate(X, Y)
'''

def plot_conf_mat(conf_mat):
    labels = ['malware', 'goodware']
    cm = conf_mat
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return


plot_result()
print "end"
