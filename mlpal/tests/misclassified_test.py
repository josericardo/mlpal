#!/usr/bin/env python

import unittest
import numpy as np
from ..misclassified import find_errors
from utils import run_mlpal
from sklearn.cross_validation import KFold
from sklearn.dummy import DummyClassifier

X = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
y = np.array([1, 1, 0, 0])

folds = KFold(len(y), n_folds=2)
# folds will always be
# [2, 3], [0, 1]
# [0, 1], [2, 3]

class FakeClf:
    def __init__(self, prediction):
        """ A classifier that always predicts `prediction` """
        self.prediction = prediction
    def fit(self, X, y): pass
    def predict(self, X): return [self.prediction]*len(X)

class MisclassifiedTest(unittest.TestCase):
    def test_false_positives_are_found(self):
        clf = FakeClf(1)
        errors = find_errors(clf, X, y, folds)
        fpos, fneg = errors[(0, 1)], errors[(1, 0)]
        self.assertEquals(0, len(fneg)) # clf always predicts as 1
        self.assertTrue(np.array_equal(fpos, [[4, 5], [6, 7]]))

    def test_false_negatives_are_found(self):
        clf = FakeClf(0)
        errors = find_errors(clf, X, y, folds)
        fpos, fneg = errors[(0, 1)], errors[(1, 0)]
        self.assertEquals(0, len(fpos)) # clf always predicts as 0
        self.assertTrue(np.array_equal(fneg, [[0, 1], [2, 3]]))

    def test_misclassified_task_is_ok(self):
        self.assertEqual(None, run_mlpal('misclassified -f --cv=2'), "misclassified task with force is failing")
        self.assertEqual(None, run_mlpal('misclassified --cv=2'), "misclassified task without force is failing")
