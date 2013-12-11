#!/usr/bin/env python

import logging

def logger_for(config, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(config.log_to)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(name)s#%(funcName)s %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    if not config.q:
        logger.addHandler(ch)

    return logger
