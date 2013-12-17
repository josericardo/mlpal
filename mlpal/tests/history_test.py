#!/bin/env python
# coding=utf-8


import unittest
from ..history import History


def create_a_history():
    h = History()
    entry = h.new()
    h.append(entry)

class HistoryTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        History().erase()

    def test_starts_a_history(self):
        create_a_history()
        h2 = History()
        self.assertEqual(1, len(h2))

    def test_finds_an_individual_entry(self):
        create_a_history()
        h = History()
        self.assertTrue(h.get(0) is not None)
