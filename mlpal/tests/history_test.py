#!/bin/env python
# coding=utf-8


import unittest
from ..history import History


class HistoryTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        History().erase()

    def test_starts_a_history(self):
        h = History()
        entry = h.new()
        h.append(entry)

        h2 = History()
        self.assertEqual(1, len(h2))
