#!/usr/bin/env python

import unittest
import numpy as np

from collections import namedtuple
from mlpal.feature_extractors import ColumnExtractor, ColsExtractor

Row = namedtuple('Row', ['trecho', 'palavra', 'Child'])
Child = namedtuple('Child', ['prop'])

X = [Row('a', 'b', Child('w')), Row('c', 'd', Child('z'))]

class ColumnExtractorTest(unittest.TestCase):
    def test_transforms_fetching_only_one_column(self):
        ce = ColumnExtractor(['trecho'])
        actual = ce.transform(X)
        expected = [['a'], ['c']]
        self.assertEquals(expected, actual)

    def test_transforms_fetching_two_columns(self):
        ce = ColumnExtractor(['trecho', 'palavra'])
        actual = ce.transform(X)
        expected = [['a', 'b'], ['c', 'd']]
        self.assertEquals(expected, actual)

    def test_transforms_fetching_one_attribute_column(self):
        ce = ColumnExtractor(['Child.prop'])
        actual = ce.transform(X)
        expected = [['w'], ['z']]
        self.assertEquals(expected, actual)

class ColsExtractorTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0, 1, 2],
            [3, 4, 5]
        ])

    def test_extracts_one_col(self):
        ce = ColsExtractor([1])
        self.assertEqual([[1], [4]], ce.transform(self.X))

    def test_extracts_more_than_one_col(self):
        ce = ColsExtractor([0, 2])
        features = ce.transform(self.X)

        expected = [
            [0, 2],
            [3, 5]
        ]

        self.assertTrue(np.array_equal(expected, features))

    def test_extracts_and_flattens(self):
        ce = ColsExtractor([0], flat=True)
        features = ce.transform(self.X)

        expected = [0, 3]
        self.assertTrue(np.array_equal(expected, features))
