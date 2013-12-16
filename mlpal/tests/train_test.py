#!/bin/env python
# encoding=utf-8

import unittest
import utils

class Test(unittest.TestCase):
    def test_train_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('train'), "train is failing")
