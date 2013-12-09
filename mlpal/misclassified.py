#!/usr/bin/env python
# encoding: utf-8

# IMPORTANT run this on the training set

# The evaluation must be on the cross validation set
# cross-validate and build a set with the unique
# examples that were misclassified
def print_errors(clf, data_source):
    X_train, y_train = data_source.train_data()

    i = 0
    y_pred = clf.predict(X_train)
    missed = 0

    for x in X_train:
        if y_train[i] != y_pred[i]:
            # if not stupid(x):
            missed += 1
            print("(%d, %d) = %s" % (y_train[i], y_pred[i], x))
        i += 1

    print("Missed %d of %d" % (missed, len(X_train)))

