#!/usr/bin/env python

import sys
import joblib

from sklearn.pipeline import Pipeline

from train import Trainer
from search import Searcher
from benchmark import Benchmarker
from misclassified import print_misclassified
from learning_curves import plot_lcs
from pca import plot_pca
from history import History
from importlib import import_module
from config import Config

def run_learning_curves(rt):
    X, y = rt.data_source.train_data()
    plot_lcs(rt.spec, X, y, rt.config)

def run_benchmark(rt):
    args = rt.config
    print("Loading classifier: %s" % args.clfpath)
    classifier = joblib.load(args.clfpath)
    # check if the user has not defined one?
    args.log_to = 'benchmark.log'
    benchmarker = Benchmarker(args, rt.data_source, classifier)
    benchmarker.run()

def run_plot_pca(rt):
    X, y = rt.data_source.train_data()

    clf = rt.spec.training_classifier()
    if isinstance(clf, Pipeline):
        X = clf.fit_transform(X, y)

    if X.shape[1] < 2:
        raise RuntimeError('You must have at least 2 features to generate a PCA plot')

    plot_pca(X, y)

def run_train(rt):
    trainer = Trainer(rt.config, rt.data_source,
        rt.spec.training_classifier())
    trainer.run()

def run_search(rt):
    searcher = Searcher(rt.config, rt.data_source)
    searcher.fit(rt.spec.gridsearch_pipelines())

def run_misclassified(rt):
    print_misclassified(rt.config, rt.spec.training_classifier(), rt.data_source)

class MLPalRuntime:
    pass

def run(args):
    setup = import_module(args.setup)

    runtime = MLPalRuntime()
    runtime.info = History().new()
    runtime.data_source = setup.DataSource(args)
    runtime.spec = setup.LearningSpec()
    runtime.config = args

    task = args.task
    task_function = getattr(sys.modules[__name__], 'run_%s' % task)
    task_function(runtime)
