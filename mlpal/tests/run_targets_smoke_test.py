#!/usr/bin/env python

import unittest
import os
import utils


class RunTargetsSmokeTest(unittest.TestCase):
    def test_learning_curves_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('learning_curves -n 100 --cv=2 --points=2'), "learning_curves is failing")

    def test_benchmark_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('train'), "train is failing")
        self.assertEqual(None, utils.run_mlpal('benchmark --clfpath=dumps/last.dump'), "benchmark is failing")

    def test_peek_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('peek -n 10'), "peek is failing")
