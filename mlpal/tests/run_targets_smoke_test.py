#!/usr/bin/env python

import unittest
import os
from utils import exit_code_of

def mlpal(task):
    cmd = "bin/mlpal --tests %s mlpal.tests.dummy_setup" % (task)
    print("\nRunning: %s" % cmd)
    return cmd

class RunTargetsSmokeTest(unittest.TestCase):
    def test_train_is_ok(self):
        self.assertEqual(None, exit_code_of(mlpal('train')), "train is failing")

    def test_search_is_ok(self):
        self.assertEqual(None, exit_code_of(mlpal('search --cv=3 -j 2 --random-state=42')), "search is failing")

    def test_plot_pca_is_ok(self):
        self.assertEqual(None, exit_code_of(mlpal('plot_pca')), "plot_pca is failing")

    def test_learning_curves_is_ok(self):
        self.assertEqual(None, exit_code_of(mlpal('learning_curves -n=100')), "learning_curves is failing")

    def test_benchmark_is_ok(self):
        self.assertEqual(None, exit_code_of(mlpal('train')), "train is failing")
        self.assertEqual(None, exit_code_of(mlpal('benchmark --clfpath=dumps/last.pickle')), "benchmark is failing")

    def test_misclassified_is_ok(self):
        self.assertEqual(None, exit_code_of(mlpal('misclassified --cv=3')), "misclassified is failing")
