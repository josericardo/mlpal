#!/usr/bin/env python

import dummy_setup
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

class DataSource(dummy_setup.DataSource):
    pass

class LearningSpec(dummy_setup.LearningSpec):
    def training_classifier(self):
        return Pipeline([('clf', LinearSVC())])
