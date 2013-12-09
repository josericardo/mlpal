#!/usr/bin/env python

import os
from sklearn.dummy import DummyClassifier
import numpy as np
import random
import ipdb

from ..base_datasource import BaseDataSource

class Split:
    pass

rows = 500
# a rows x 2 matrix
a_dataset = np.array([
    range(0, rows),
    range(0, rows),
    sorted([0]*(rows/2) + [1]*(rows/2), key=os.urandom)
]).T

class DataSource(BaseDataSource):
    def __init__(self, args):
        pass

    def slice_for_tests(self, start, end):
        return a_dataset

    def get_test_size(self):
        return len(a_dataset)

    def raw_to_matrix(self, data):
        return data

    def raw_train(self): return a_dataset
    def raw_validation(self): return a_dataset
    def raw_test(self): return a_dataset


class LearningSpec:
    def training_classifier(self):
        return DummyClassifier()

    def gridsearch_pipelines(self):
        return {'test': {'pipeline': self.training_classifier(), 'params': {}}}
