#!/usr/bin/env python

import json
import logging
import os
from utils import dict_merge
from datetime import datetime

class Config:
    def __init__(self, config):
        default_config = {
            'train_size': 10000,
            'validation_size': 10000,
            'test_size': 10000,
            'bias': 0.3,
            'debug': True,
            'log_file': 'lastrun-%s.log' % datetime.now().strftime('%Y%m%d_%H%M%S'),
            'n_jobs': 1,
            'log_to_stdout': False
        }

        merged_config = dict_merge(default_config, self.load_config())
        merged_config = dict_merge(merged_config, config)
        self.__dict__.update(**merged_config)

    def load_config(self):
        name = 'config.json'
        if os.path.exists(name):
            print("%s was found, loading." % name)
            return json.load(open(name))
        return {}

    def logger_for(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s %(name)s#%(funcName)s %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        if self.log_to_stdout:
            logger.addHandler(ch)

        return logger
