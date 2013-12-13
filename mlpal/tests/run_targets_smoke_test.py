#!/usr/bin/env python

import unittest
import os
import utils


class RunTargetsSmokeTest(unittest.TestCase):
    def test_train_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('train'), "train is failing")

    def test_search_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('search --cv=3 -j 2 --random-state=42'), "search is failing")

    def test_plot_pca_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('plot_pca'), "plot_pca is failing")

    def test_learning_curves_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('learning_curves -n=100 --cv=2 --points=2'), "learning_curves is failing")

    def test_benchmark_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('train'), "train is failing")
        self.assertEqual(None, utils.run_mlpal('benchmark --clfpath=dumps/last.dump'), "benchmark is failing")

