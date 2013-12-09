#!/usr/bin/env python

import unittest
from mlpal.confusion_matrix import ConfusionMatrix

class ConfusionMatrixTest(unittest.TestCase):
    def test_can_report_itself_as_a_string(self):
        raw_cm = [
            [9, 10],
            [3, 10]
        ]
        str = ConfusionMatrix(raw_cm).as_str()
        self.assertTrue(str.find('False positives: 10 (50.00%') != -1)
        self.assertTrue(str.find('False negatives: 3 (25.00%') != -1)
