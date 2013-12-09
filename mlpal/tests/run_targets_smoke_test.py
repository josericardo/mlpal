#!/usr/bin/env python

import unittest
import os
from utils import exit_code_of

def mlpal(task, params=''):
    return "bin/mlpal --tests %s mlpal.tests.dummy_setup %s" % (task, params)

class RunTargetsSmokeTest(unittest.TestCase):
    def test_run_targets_smoke_test(self):
        self.assertEqual(None, exit_code_of(mlpal('train')), "train is failing")
        self.assertEqual(None, exit_code_of(mlpal('search')), "search is failing")
        self.assertEqual(None, exit_code_of(mlpal('benchmark', '--clfpath=dumps/last.pickle')), "benchmark is failing")
        self.assertEqual(None, exit_code_of(mlpal('learning_curves', '-n=100')), "learning_curves is failing")
        self.assertEqual(None, exit_code_of(mlpal('plot_pca')), "plot_pca is failing")
        # TODO self.assertEqual(None, exit_code_of(mlpal('misclassified')), "misclassified is failing")
