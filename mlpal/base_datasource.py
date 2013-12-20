#!/usr/bin/env python

import numpy as np
from base_learning_spec import LearningSetupIsBroken

def no(method):
    msg = "DataSource#{method} must be defined. See BaseDataSource#{method} documentation."
    error = msg.format(method=method)
    raise LearningSetupIsBroken(error)


class BaseDataSource:
    """ DataSource interface, all code should interact only with these methods """
    def __init__(self, args):
        pass
    def train_data(self):
        """
        Train set accessor

        Returns
        -------
        (X, y): tuple
        """
        return self.Xy(self.raw_train())

    def test_data(self):
        """
        Test set accessor

        Returns
        -------
        iterator
            (X, y) tuples iterator
        """
        return (self.Xy(subset) for subset in self.raw_test())

    def Xy(self, raw_data):
        """
        Converts raw data into X and y that can be used by the scikit-learn
        classes.

        Returns
        -------
        (X, y): tuple
            X and y must be numpy arrays
        """
        no('Xy')

    def raw_train(self):
        """
        Raw train set accessor

        Returns
        -------
        train set
            data in a format that self#Xy understands
        """
        no('raw_train')

    def raw_test(self):
        """
        Raw test set accessor

        Returns
        -------
        raw test set iterator
            the benchmark will evaluate each subset (item in the iterator)
            in parallel.
            Important: each element must be in the format that
            self#Xy understands
        """
        no('raw_test')
