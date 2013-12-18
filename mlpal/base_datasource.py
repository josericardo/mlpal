#!/usr/bin/env python

import numpy as np
from base_learning_spec import LearningSetupIsBroken

def no(method):
    msg = "DataSource#{method} must be defined. See BaseDataSource#{method} documentation."
    error = msg.format(method=method)
    raise LearningSetupIsBroken(error)


class BaseDataSource:
    """ DataSource interface, all code should interact only with these methods """
    def train_data(self):
        return self.Xy(self.raw_train())

    def test_data(self):
        return self.Xy(self.raw_test())

    def testing_slice(self, start, end):
        return self.Xy(self.slice_for_tests(start, end))

    def Xy(self, raw_data):
        no('Xy')

    def raw_train(self): no('raw_train')
    def raw_test(self): no('raw_test')
    def slice_for_tests(self, start, end): no('slice_for_tests')
    def get_test_size(self): no('get_test_size')
