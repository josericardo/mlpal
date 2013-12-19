#!/usr/bin/env python

import os
from sklearn.dummy import DummyClassifier
import numpy as np
import random
from itertools import izip

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

    def Xy(self, data):
        return data[:,:2], data[:,2]

    def raw_train(self):
        return a_dataset

    def raw_test(self):
        # if you don't want to benchmark in parallel
        # return a list of one element, eg.: [a_dataset]
        slice_edges = [0, 100, 200, 300, 400, 500]
        slices = izip(slice_edges[:-1], slice_edges[1:])
        return (a_dataset[begin:end] for (begin, end) in slices)


class LearningSpec:
    def training_classifier(self):
        return DummyClassifier()

    def gridsearch_pipelines(self):
        return {'test': {'pipeline': self.training_classifier(), 'params': {}}}
