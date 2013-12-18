#!/usr/bin/env python

import numpy as np

class ClassifierAsFeature:
    """Helper to use the output of a classifier as a feature

    Params
    ------

    classifier: scikit-learn classifier
        Its predict_proba method will be used as this object's #transform
        method
    trained: bool
        If False, `classifier#fit` will be called when this object's #fit is
        called
    """
    def __init__(self, classifier, trained=False):
        self.clf = classifier
        self.trained = trained

    def fit(self, X, y=None):
        if not self.trained:
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
        return {'classifier': self.clf, 'trained': self.trained}

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self.get_params()))

class BaseFeatureExtractor:
    def transform(self, X, y=None):
        return [self.map(x) for x in X]

    def map(self, x):
        raise RuntimeError('Your Feature Extractor must define the #map method')

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        raise RuntimeError('#get_params must be defined. Your Feature Extractor must tell scikit what params it receives in its constructor')

