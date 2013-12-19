#!/bin/env python
# coding=utf-8

import unittest
import utils

class LearningCurvesTest(unittest.TestCase):
    def test_learning_curves_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('learning_curves -n 100 --cv=2 --points=2'), "learning_curves is failing")

