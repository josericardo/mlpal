#!/bin/env python
# coding=utf-8

import os
import unittest
import sys
from collections import namedtuple
from .. import runner
from ..history import History

FAKE_INFO_VALUE = "fake value"
FAKE_INFO_KEY = "fake key"

def fake_task(rt):
    """Simulates a task that wants to add info to history"""
    rt.info[FAKE_INFO_KEY] = FAKE_INFO_VALUE

setattr(sys.modules['mlpal.runner'], 'run_fake_task', fake_task)

class FakeSetup:
    def DataSource(self, args): pass
    def LearningSpec(self): pass

class FakeArgs(object):
    pass

class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.history_file = 'runner_test_history.json'

    def tearDown(self):
        os.remove(self.history_file)

    def test_saves_history(self):
        args = FakeArgs()
        args.task = 'fake_task'
        args.history_id = self.history_file.replace('.json', '')
        runner.run(args, setup=FakeSetup())

        history = History(id=args.history_id)
        self.assertEqual(FAKE_INFO_VALUE, history.get(0)[FAKE_INFO_KEY])
