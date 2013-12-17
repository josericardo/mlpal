#!/bin/env python
# coding=utf-8

import unittest
import utils


class SearchTest(unittest.TestCase):
    def test_search_is_ok(self):
        self.assertEqual(None, utils.run_mlpal('search --cv=3 -j 2 --random-state=42'), "search is failing")


