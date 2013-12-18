#!/usr/bin/env python

import unittest
from mlpal.sklearn_helper import *
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

X = [[0, 1], [0, 1], [1, 0], [1, 0]]
y = [1, 1, 0, 0]

class ClassifierAsFeatureTest(unittest.TestCase):
    def test_it_is_scikit_learn_compliant_using_predict_proba(self):
        pipe = Pipeline([
            ('caf', ClassifierAsFeature(MultinomialNB())),
            ('clf', MultinomialNB())
        ])

        pipe.fit(X, y)
        self.assertTrue(len(pipe.predict(X)) > 0)

    def test_it_defines_a_useful_repr(self):
        clf = ClassifierAsFeature(MultinomialNB())
        self.assertTrue(repr(clf).index('MultinomialNB') > 0)
