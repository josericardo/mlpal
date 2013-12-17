#!/usr/bin/env python

import os

def exit_code_of(cmd):
    process = os.popen(cmd)
    process.read()
    return process.close()

def mlpal(task, setup):
    cmd = "bin/mlpal %s %s" % (task, setup)
    print("\nRunning: %s" % cmd)
    return cmd

def run_mlpal(task, setup='mlpal.tests.dummy_setup'):
    return exit_code_of(mlpal(task, setup))

