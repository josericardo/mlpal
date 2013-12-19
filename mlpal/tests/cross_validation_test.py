#!/bin/env python
# coding=utf-8

import unittest
from .dummy_setup import DataSource, LearningSpec
from .. import cross_validation as cv

X, y = DataSource(None).train_data()
clf = LearningSpec().training_classifier()

class Config:
    pass

class CrossValidationTest(unittest.TestCase):
    def setUp(self):
        config = Config()
        config.scoring = 'f1'
        config.cv = 3
        config.random_state = 0

        self.config = config

    def test_cross_validates_serially(self):
        self.config.j = 1
        self.assertEquals(self.config.cv, len(cv._serial_cv_scores(clf, X, y, self.config)))

    def test_cross_validates_in_parallel(self):
        self.config.j = 2
        self.assertEquals(self.config.cv, len(cv._parallel_cv_scores(clf, X, y, self.config)))
