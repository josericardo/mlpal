#!/bin/env python
# coding=utf-8

import unittest
import utils


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('train'), "train is failing")
        self.assertEqual(None, utils.run_mlpal('benchmark --clfpath=dumps/last.dump'), "benchmark is failing")

