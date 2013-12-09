#!/usr/bin/env python
# encoding: utf-8

from sklearn.cross_validation import StratifiedShuffleSplit

# IMPORTANT run this on the training set

# The evaluation must be on the cross validation set
# cross-validate and build a set with the unique
# examples that were misclassified
def find_errors(clf, X, y, splits):
    false_positives = []
    false_negatives = []

    for train_index, test_index in splits:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        for i, x in enumerate(X_test):
            if y_pred[i] == 0 and y_test[i] == 1:
                false_negatives.append(x)
            elif y_pred[i] == 1 and y_test[i] == 0:
                false_positives.append(x)

    return false_positives, false_negatives

def print_misclassified(config, clf, data_source):
    X, y = data_source.train_data()

    sss = StratifiedShuffleSplit(y, n_iter=config.cv)
    false_pos, false_neg = find_errors(clf, X, y, sss)

    print("%d false negatives: " % len(false_neg))
    print(false_neg)

    print("%d false positives: " % len(false_pos))
    print(false_pos)
