#!/usr/bin/env python

import os

def exit_code_of(cmd):
    process = os.popen(cmd)
    process.read()
    return process.close()

def mlpal(task):
    cmd = "bin/mlpal --tests %s mlpal.tests.dummy_setup" % (task)
    print("\nRunning: %s" % cmd)
    return cmd

def run_mlpal(task):
    return exit_code_of(mlpal(task))

