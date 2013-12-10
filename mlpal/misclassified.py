#!/usr/bin/env python
# encoding: utf-8

from sklearn.cross_validation import StratifiedShuffleSplit
import pickle

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

def print_misclassified(config, clf, data_source):
    X, y = data_source.train_data()

    sss = StratifiedShuffleSplit(y, n_iter=config.cv)
    errors = find_errors(clf, X, y, sss)
    false_pos, false_neg = errors[(0, 1)], errors[(1, 0)]

    dump_path = 'misclassifications.pickle'
    print("Dumping misclassifications to %s..." % dump_path),
    pickle.dump(errors, open(dump_path, 'wb'))
    print("done.")

    _print({'id': 'FN', 'class': 'False Negatives', 'X': false_neg})
    _print({'id': 'FP', 'class': 'False Positives', 'X': false_pos})
