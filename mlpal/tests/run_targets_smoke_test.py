#!/usr/bin/env python

import unittest
import os
import utils


class RunTargetsSmokeTest(unittest.TestCase):
    def test_peek_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('peek -n 10'), "peek is failing")
