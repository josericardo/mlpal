#!/usr/bin/env python

import numpy as np

class ClassifierAsFeature:
    """ Helper to use the output of a classifier as a feature """
    def __init__(self, classifier):
        self.clf = classifier

    def fit(self, X, y=None):
        self.clf.fit(X, y)
        return self

    def transform(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, X, y):
        return self.clf.score(X, y)

    def get_params(self, deep=True):
        # TODO return the params of all children
        return {'classifier': self.clf}

class BaseFeatureExtractor:
    def transform(self, X, y=None):
        return [self.map(x) for x in X]

    def map(self, x):
        raise RuntimeError('Your Feature Extractor must define the #map method')

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        raise RuntimeError('#get_params must be defined. Your Feature Extractor must tell scikit what params it receives in its constructor')

