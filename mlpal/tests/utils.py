#!/usr/bin/env python

import os

def exit_code_of(cmd):
    process = os.popen(cmd)
    process.read()
    return process.close()

