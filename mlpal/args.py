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

    if hasattr(args, 'setup'):
        args.setup = _normalize_setup_path(args.setup)

    args.mlpal_version = __version__

    return args

def parse_args(args=None):
    defaults = defaultdict(lambda: None)
    config = _load_config()

    for k, v in config.iteritems():
        defaults[k] = v

    parser = generate_parser(defaults)
    parsed_args = parser.parse_args() if not args else parser.parse_args(args)
    return post_process(config, parsed_args)

def generate_parser(defaults):
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    subparsers = parser.add_subparsers(title="subcommands", dest='task')

    common_parser = gen_common_args_parser(defaults)
    ml_parser = machine_learning_args_parser(common_parser, defaults)

    add_mlpal_tasks_to(subparsers, common_parser)
    add_machine_learning_tasks_to(subparsers, ml_parser, defaults)

    return parser

def gen_common_args_parser(defaults):
    common_parser = argparse.ArgumentParser()

    common_parser.add_argument("-d", help="Launch debugger on exception", action='store_true')
    common_parser.add_argument("-q", help="Less verbosity on STDOUT",
        action='store_true', default=defaults.get('q', False))
    common_parser.add_argument("-f", help="Force", action='store_true')
    common_parser.add_argument("--desc",  default='', type=str,
            help="Textual description of this run (goes to history)")
    common_parser.add_argument("--history-id", type=str,
        help="History file's id",
        default=defaults.get('history_id', 'history'))

    return common_parser

def add_mlpal_tasks_to(subparsers, common_parser):
    new_setup_parser = subparsers.add_parser('new_setup', parents=[common_parser],
            help='Generates a new setup file.',
            conflict_handler='resolve')
    new_setup_parser.add_argument("new_setup_id", help="New setup id.", default='new_setup')

def add_machine_learning_tasks_to(subparsers, ml_parser, defaults):
    subparsers.add_parser('plot_pca', parents=[ml_parser],
            help='Plot the data reduced to 2-dimensions',
            conflict_handler='resolve')

    subparsers.add_parser('peek', parents=[ml_parser],
            help='A peek at the data that is sent to the classifier.',
            conflict_handler='resolve')

    misclassified_parser = subparsers.add_parser('misclassified', parents=[ml_parser],
            help='Prints the examples that were misclassified during Cross Validation',
            conflict_handler='resolve')

    misclassified_parser.add_argument("--cols", type=str, default=defaults.get('cols'),
            help="List of X cols to display. Eg.: '1,2,4' or '2'")

    train_search_parser = argparse.ArgumentParser(parents=[ml_parser], conflict_handler='resolve')

    train_search_parser.add_argument("--scoring", type=str, default=defaults.get('scoring', 'f1'),
        help="Scoring function")

    train_search_parser.add_argument("--random-state", type=int,
        default=defaults['random_state'],
        help="Pseudo random number generator state")

    train_parser = subparsers.add_parser('train', parents=[train_search_parser],
            help='Train a classifier',
            conflict_handler='resolve')

    search_parser = subparsers.add_parser('search', parents=[train_search_parser],
            help='Grid Search',
            conflict_handler='resolve')

    benchmark_parser = subparsers.add_parser('benchmark', parents=[ml_parser],
            help='Benchmark a classifier on the test set.',
            conflict_handler='resolve')

    benchmark_parser.add_argument("--clfpath", default='dumps/last.dump',
        help="Serialized classifier path (benchmarks only)")

    lc_parser = subparsers.add_parser('learning_curves', parents=[ml_parser],
            help='Plot learning curves',
            conflict_handler='resolve')

    lc_parser.add_argument("--begin", type=int,
        help="The size of the smallest training set used to plot the learning curves",
        default=defaults.get('begin', 30))

    lc_parser.add_argument("--points", type=int,
        help="Number of points in the learning curves",
        default=defaults.get('points', 8))

    lc_parser.add_argument("--space", type=str, choices=['log', 'linear'],
        help="How are the points going to be spread in the space?",
        default=defaults.get('space', 'log'))

def machine_learning_args_parser(common_parser, defaults):
    ml_parser = argparse.ArgumentParser(parents=[common_parser], conflict_handler='resolve')
    ml_parser.add_argument("setup", help="Python module with the running definitions")
    ml_parser.add_argument("-n", help="Sample size", type=int, default=defaults['n'])
    ml_parser.add_argument("-j", help="# of jobs", type=int, default=defaults.get('j', 1))
    ml_parser.add_argument("--cv", type=int, default=defaults.get('cv', 10),
        help="Number of cv iterations")

    ml_parser.add_argument("-o", help="Output files prefix.", type=str, default=defaults['o'])

    default_log_name = 'lastrun-%s.log' % datetime.now().strftime('%Y%m%d_%H%M%S')
    ml_parser.add_argument("--log-to", type=str, help="Log file path",
        default=defaults.get('log_to', default_log_name))

    return ml_parser
