#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import re
import yaml

from datetime import datetime
from collections import defaultdict
from .version import __version__

def _load_config():
    # TODO read $MLPAL_TESTS
    name = 'config.yaml'
    if os.path.exists(name):
        print("%s was found, loading." % name)
        return yaml.load(open(name))
    return {}

def fill_user_space_config(config, args):
    # things that are defined in the config but are not known params
    # are considered user-space configuration
    for k, v in config.iteritems():
        if not hasattr(args, k):
            setattr(args, k, v)

    return args

def _normalize_setup_path(path):
    new_path = re.sub(r'\.py$', '', path)
    return new_path.replace(os.sep, '.')

def post_process(config, args):
    args = fill_user_space_config(config, args)
    args.setup = _normalize_setup_path(args.setup)
    args.mlpal_version = __version__

    return args

def parse_args():
    defaults = defaultdict(lambda: None)
    config = _load_config()

    for k, v in config.iteritems():
        defaults[k] = v

    tasks = ['train', 'search', 'benchmark',
             'learning_curves', 'plot_pca', 'misclassified']

    parser = argparse.ArgumentParser()

    parser.add_argument("task", choices=tasks, help="The task to be run")
    parser.add_argument("setup", help="Python module with the running definitions")
    parser.add_argument("-d", help="Launch debugger on exception", action='store_true')
    parser.add_argument("-n", help="Sample size", type=int, default=defaults['n'])
    parser.add_argument("-j", help="# of jobs", type=int, default=defaults.get('j', 1))
    parser.add_argument("-o", help="Output files prefix.", type=str, default=defaults['o'])
    parser.add_argument("-f", help="Force", action='store_true')
    parser.add_argument("-q", help="Less verbosity on STDOUT",
        action='store_true', default=defaults.get('q', False))

    parser.add_argument("--clfpath", default='dumps/last.dump',
        help="Serialized classifier path (benchmarks only)")

    parser.add_argument("--cv", type=int, default=defaults.get('cv', 10),
        help="Number of cv iterations")

    parser.add_argument("--scoring", type=str, default=defaults.get('scoring', 'f1'),
        help="Scoring function")

    parser.add_argument("--random-state", type=int,
        default=defaults['random_state'],
        help="Pseudo random number generator state")

    default_log_name = 'lastrun-%s.log' % datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.add_argument("--log-to", type=str, help="Log file path",
        default=defaults.get('log_to', default_log_name))

    # begin learning curves
    parser.add_argument("--begin", type=int,
        help="The size of the smallest training set used to plot the learning curves",
        default=defaults.get('begin', 30))

    parser.add_argument("--points", type=int,
        help="Number of points in the learning curves",
        default=defaults.get('points', 8))

    parser.add_argument("--space", type=str, choices=['log', 'linear'],
        help="How are the points going to be spread in the space?",
        default=defaults.get('space', 'log'))
    # end learning curves

    parser.add_argument("--history-id", type=str,
        help="History file's id",
        default=defaults.get('history_id', 'history'))

    return post_process(config, parser.parse_args())
