#!/usr/bin/env python

import numpy as np
from sklearn_helper import BaseFeatureExtractor

class ColumnExtractor:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [[self.deep_getattr(x, column) for column in self.columns] for x in X]

    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def deep_getattr(self, obj, column):
        """ Fetches properties from child objects 

        Works for:

        deep_gettattr(obj, 'attribute')
        deep_gettattr(obj, 'child.attribute')
        """
        if column.find('.') != -1:
            child, attr = column.split('.')
            child_obj = getattr(obj, child)
            return None if child_obj is None else getattr(child_obj, attr)

        return getattr(obj, column)

    def get_params(self, deep=True):
        return {'columns': self.columns}


class GetFirst(BaseFeatureExtractor):
    def map(self, x):
        return x[0]

    def get_params(self, deep=True):
        return {}


class ColsExtractor(BaseFeatureExtractor):
    """ Extracts columns from a matrix 

    Params
    ------

    cols: list of column indices
    astype: if defined, will be used to convert
            the new matrix data. Eg.: int, str
    """
    def __init__(self, cols, astype=None):
        self.cols = cols
        self.astype = astype

    def map(self, x):
        if self.astype:
            return x[self.cols].astype(self.astype)
        return x[self.cols]

    def get_params(self, deep=True):
        return {'cols': self.cols, 'astype': self.astype}
