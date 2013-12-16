#!/bin/env python
# encoding: utf-8

import unittest
import utils


class Test(unittest.TestCase):

    def test_plot_pca_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('plot_pca'), "plot_pca is failing")
        self.assertEqual(None, utils.run_mlpal('plot_pca', setup='mlpal.tests.dummy_setup_with_pipeline'), "plot_pca is failing")


