#!/usr/bin/env python

import unittest
import os

from ..base_learning_spec import BaseLearningSpec
from sklearn.dummy import DummyClassifier
from dummy_setup import DataSource as DummyDS
import utils


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
    SETUP = 'mlpal.tests.base_learning_spec_test'

    def test_BaseLearningSpec_can_be_used_for_training(self):
        train = 'train  --cv=2'
        self.assertEqual(None, utils.run_mlpal(train, self.SETUP), "make train is failing")

    def test_BaseLearningSpec_can_be_used_for_searching(self):
        search = 'search --cv=2'
        self.assertEqual(None, utils.run_mlpal(search, self.SETUP), "make search is failing")
