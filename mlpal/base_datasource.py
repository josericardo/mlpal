#!/usr/bin/env python

import numpy as np
from base_learning_spec import LearningSetupIsBroken

def no(method):
    msg = "DataSource#{method} must be defined. See BaseDataSource#{method} documentation."
    error = msg.format(method=method)
    raise LearningSetupIsBroken(error)


class BaseDataSource:
    """ DataSource interface, all code should interact only with these methods """
    def _X_and_y(self, data):
        matrix = self.raw_to_matrix(data)
        X = [m[:-1] for m in matrix]
        y = [m[-1] for m in matrix]
        return np.array(X), np.array(y)

    def train_data(self):
        return self._X_and_y(self.raw_train())

    # TODO remove, we use only cross-validation by now
    def validation_data(self):
        return self._X_and_y(self.raw_validation())

    def test_data(self):
        return self._X_and_y(self.raw_test())

    def testing_slice(self, start, end):
        return self._X_and_y(self.slice_for_tests(start, end))

    def raw_to_matrix(self, data):
        """ A normalizer, makes the BaseDataSource generic

        Must return a matrix in the form:

        [
            [ x11, x12, ..., x1m, y11],
            [ x21, x22, ..., x2m, y21],
            ...
            [ xn1, xn2, ..., xnm, yn1],
        ]

        where,

        x = features
        y = labels

        """
        no('raw_to_matrix')

    def raw_train(self): no('raw_train')
    def raw_validation(self): no('raw_validation')
    def raw_test(self): no('raw_test')
    def slice_for_tests(self, start, end): no('slice_for_tests')
    def get_test_size(self): no('get_test_size')
