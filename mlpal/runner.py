#!/usr/bin/env python

import sys
import pickle

from train import Trainer
from search import Searcher
from benchmark import Benchmarker
from misclassified import print_misclassified
from learning_curves import plot_lcs
from pca import plot_pca
from importlib import import_module
from config import Config

def run_learning_curves(args, spec, data_source):
    X, y = data_source.train_data()
    plot_lcs(spec, X, y, args)

def run_benchmarks(args, data_source):
    print("Loading classifier: %s" % args.clfpath)
    classifier = pickle.load(open(args.clfpath, "rb"))
    config = Config({'log_file': 'benchmark.log'})
    benchmarker = Benchmarker(args, config, data_source, classifier)
    benchmarker.run()

def run(args):
    setup = import_module(args.setup)
    task = args.task
    data_source = setup.DataSource(args)
    spec = setup.LearningSpec()

    if task == 'train':
        trainer = Trainer(args, Config({}), data_source, spec.training_classifier())
        trainer.run()
    elif task == 'search':
        searcher = Searcher(args, Config({}), data_source)
        searcher.fit(spec.gridsearch_pipelines())
    elif task == 'benchmark':
        run_benchmarks(args, data_source)
    elif task == 'learning_curves':
        run_learning_curves(args, spec, data_source)
    elif task == 'plot_pca':
        X, y = data_source.train_data()
        plot_pca(X, y)
    elif task == 'misclassified':
        print_misclassified(args, spec.training_classifier(), data_source)
