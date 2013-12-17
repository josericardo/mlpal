#!/bin/env python
# coding=utf-8


import unittest
import utils
import os

SETUP_ID = 'my_new_project'
SETUP_FILE = 'my_new_project.py'

class Test(unittest.TestCase):
    def tearDown(self):
        os.remove(SETUP_FILE)

    def test_generates_new_setup(self):
        utils.run_mlpal('new_setup', setup=SETUP_ID)
        self.assertTrue(os.path.isfile(SETUP_FILE))

        with open(SETUP_FILE, 'r') as f:
            content = f.read()
            self.assertTrue(content.find('DataSource') > -1)
            self.assertTrue(content.find('LearningSpec') > -1)
