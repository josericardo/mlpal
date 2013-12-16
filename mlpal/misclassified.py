#!/usr/bin/env python
# coding=utf-8

from sklearn.cross_validation import StratifiedShuffleSplit
import joblib
import os

_DUMP_PATH = 'misclassifications.dump'

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

    return {(0, 1): false_positives, (1, 0): false_negatives}

def _print(data):
    print("%d %s" % (len(data['X']), data['class']))

    for x in data['X']:
        print("%s | %s\n============\n" % (data['id'], x[0]))

def _print_errors(errors):
    false_pos, false_neg = errors[(0, 1)], errors[(1, 0)]

    _print({'id': 'FN', 'class': 'False Negatives', 'X': false_neg})
    _print({'id': 'FP', 'class': 'False Positives', 'X': false_pos})

def _load_previous_classification(config):
    if config.f:
        return None

    if os.path.isfile(_DUMP_PATH):
        print("Previous data was found. Loading %s" % _DUMP_PATH)
        return joblib.load(_DUMP_PATH)

    print("No previous data found. Generating.")

def print_misclassified(config, clf, data_source):
    errors = _load_previous_classification(config)

    if not errors:
        errors = classify(config, clf, data_source)

    _print_errors(errors)

def classify(config, clf, data_source):
    X, y = data_source.train_data()
    sss = StratifiedShuffleSplit(y, n_iter=config.cv)
    errors = find_errors(clf, X, y, sss)

    print("Dumping misclassifications to %s..." % _DUMP_PATH),
    joblib.dump(errors, _DUMP_PATH, compress=1)
    print("done.")

    return errors
