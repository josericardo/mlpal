#!/bin/env python
# coding=utf-8

import numpy as np
from sklearn.metrics.scorer import _deprecate_loss_and_score_funcs
from sklearn import cross_validation

def cross_validate(clf, X, y, config):
    if config.j == 1:
        return _serial_cv_scores(clf, X, y, config)
    else:
        return _parallel_cv_scores(clf, X, y, config)

def _serial_cv_scores(clf, X, y, config):
    # sklearn's cross_val_score clones the classifier
    # because it cross validates in parallel
    # there are situations in which cloning is not desirable
    # (eg.: stacking classifiers), the serial cross-validation
    # is solution for this problem
    scorer = _deprecate_loss_and_score_funcs(scoring=config.scoring)

    sss = cross_validation.StratifiedShuffleSplit(
            y,
            test_size=0.1,
            n_iter=config.cv,
            random_state=config.random_state
    )

    scores = []

    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        scores.append(scorer(clf, X_test, y_test))

    scores = np.array(scores)

    return scores

def _parallel_cv_scores(clf, X, y, config):
    return cross_validation.cross_val_score(clf, X, y,
            cv=config.cv,
            scoring=config.scoring
    )
