#!/usr/bin/env python

import unittest
import os

from ..base_learning_spec import BaseLearningSpec
from sklearn.dummy import DummyClassifier
from dummy_setup import DataSource as DummyDS
from utils import exit_code_of

class DataSource(DummyDS):
    pass

class LearningSpec(BaseLearningSpec):
    def training_classifier(self):
        return DummyClassifier()

    def gridsearch_params(self):
        return {
            'default': {
                'strategy': ('most_frequent', 'uniform')
            }
        }

class BaseLearningSpecTest(unittest.TestCase):
    def test_LearningSpec_can_relly_on_BaseLearningSpec(self):
        self.assertEqual(None, exit_code_of('bin/mlpal --tests train mlpal.tests.base_learning_spec_test'), "make train is failing")
        self.assertEqual(None, exit_code_of('bin/mlpal --tests search mlpal.tests.base_learning_spec_test'), "make search is failing")
