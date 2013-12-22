#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import re
import yaml

from datetime import datetime
from collections import defaultdict
from .version import __version__

def load_defaults():
    def _config():
        name = 'config.yaml'
        if os.path.exists(name):
            print("%s was found, loading." % name)
            return yaml.load(open(name))
        return {}

    return defaultdict(lambda: None, _config())

defaults = load_defaults()

common_args = {
    "-d": {'help': 'Launch debugger on exception', 'action': 'store_true'},
    "-q": {'help': "Less verbosity on STDOUT", 'action': 'store_true',
           'default': defaults.get('q', False)},
    "-f": {'help': "Force", 'action': 'store_true'},
    "--desc": {'default': '', 'type': str,
        'help': "Textual description of this run (goes to history)"},
    "--history-id": {'type': str, 'help': "History file's id",
                     'default': defaults.get('history_id', 'history')}
}

default_log_name = 'lastrun-%s.log' % datetime.now().strftime('%Y%m%d_%H%M%S')

ml_args = {
    "setup": {'help': "Python module with the running definitions"},
    "-n": {'help': "Sample size", 'type': int, 'default': defaults['n']},
    "-j": {'help': "# of jobs", 'type': int, 'default': defaults.get('j', 1)},
    "--cv": {'type': int, 'default': defaults.get('cv', 10),
        'help': "Number of cv iterations"},
    "-o": {'help': "Output files prefix.", 'type': str, 'default': defaults['o']},
    "--log-to": {'type': str, 'help': "Log file path",
        'default': defaults.get('log_to', default_log_name)}
}

train_and_search_args = {
    "--scoring": {'type': str, 'default': defaults.get('scoring', 'f1'),
                  'help': "Scoring function"},
    "--random-state": {'type': int, 'default': defaults['random_state'],
                       'help': "Pseudo random number generator state"}
}

ml_parsers = {
    'plot_pca': {'help': 'Plot the data reduced to 2-dimensions'},
    'peek': {'help': 'A peek at the data that is sent to the classifier.'},
    'misclassified': {'help': 'Prints the examples that were misclassified during Cross Validation',
        'args': {
            "--cols": {'type': str, 'default': defaults.get('cols'), 'help': "List of X cols to display. Eg.: '1,2,4' or '2'"}
        }
    },
    'learning_curves': {'help': 'Plot learning curves',
        'args': {
            "--begin":  {'type': int, 'default': defaults.get('begin', 30),
                         'help': "The size of the smallest training set used to plot the learning curves"},
            "--points": {'type': int, 'default': defaults.get('points', 8),
                         'help': "Number of points in the learning curves"},
            "--space":  {'type': str, 'choices': ['log', 'linear'], 'default': defaults.get('space', 'log'),
                         'help': "How are the points going to be spread in the space?"}
        }
    },
    'benchmark': {'help': 'Benchmark a classifier on the test set.',
        'args': {
            "--clfpath":{'type': str, 'default': 'dumps/last.dump',
                         'help': "Serialized classifier path (benchmarks only)"}
        }
    },
    'train': {'help': 'Train a classifier', 'args': train_and_search_args},
    'search': {'help': 'Grid Search', 'args': train_and_search_args}
}

def parse_args(args=None):
    parser = _generate_parser()

    # `args` can be defined for testing
    parsed_args = parser.parse_args(args) if args else parser.parse_args()
    return _post_process(parsed_args)

def _generate_parser():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    subparsers = parser.add_subparsers(title="subcommands", dest='task')

    common_parser = _gen_common_args_parser()
    ml_parser = _gen_ml_args_parser(common_parser)
    _add_mlpal_tasks_to(subparsers, common_parser)
    _add_ml_tasks_to(subparsers, ml_parser, defaults)
    return parser

def _fill_user_space_config(config, args):
    # things that are defined in the config but are not known params
    # are considered user-space configuration
    for k, v in config.iteritems():
        if not hasattr(args, k):
            setattr(args, k, v)

    return args

def _normalize_setup_path(path):
    new_path = re.sub(r'\.py$', '', path)
    return new_path.replace(os.sep, '.')

def _post_process(args):
    args = _fill_user_space_config(defaults, args)

    if hasattr(args, 'setup'):
        args.setup = _normalize_setup_path(args.setup)

    args.mlpal_version = __version__
    return args

def _add_args(parser, arguments):
    for arg, options in arguments.iteritems():
        parser.add_argument(arg, **options)
    return parser

def _gen_common_args_parser():
    common_parser = argparse.ArgumentParser()
    return _add_args(common_parser, common_args)

def _gen_ml_args_parser(common_parser):
    ml_parser = argparse.ArgumentParser(parents=[common_parser], conflict_handler='resolve')
    return _add_args(ml_parser, ml_args)

def _add_mlpal_tasks_to(subparsers, common_parser):
    new_setup_parser = subparsers.add_parser('new_setup', parents=[common_parser],
            help='Generates a new setup file.',
            conflict_handler='resolve')
    new_setup_parser.add_argument("new_setup_id", help="New setup id.", default='new_setup')

def _add_ml_tasks_to(subparsers, ml_parser, defaults):
    for task, params in ml_parsers.iteritems():
        args = None
        if 'args' in params:
            args = params.pop('args')

        params['parents'] = [ml_parser]
        params['conflict_handler'] = 'resolve'
        parser = subparsers.add_parser(task, **params)

        if args:
            _add_args(parser, args)
