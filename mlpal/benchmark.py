#!/usr/bin/env python

import copy
from collections import namedtuple
import traceback
import math
from joblib import Parallel, delayed
from train import Trainer
import log_utils
import logger_factory

BenchmarkParams = namedtuple('BenchmarkParams', ['rt', 'ds', 'config', 'clf'])

def _benchmark(params, test_subset):
    """Benchmarks the classifier based on a test set slice"""
    try:
        X_test, y_test = test_subset
        clf = copy.deepcopy(params.clf)
        trainer = Trainer(params.rt, params.config, params.ds, params.clf)
        return trainer.benchmark(X_test, y_test).confusion_matrix
    except Exception as e:
        print(e)
        traceback.print_exc()

def _merge_confusion_matrices(cms_list):
    merged = [[0,0], [0,0]]

    for cm in cms_list:
        merged[0][0] += cm[0][0]
        merged[0][1] += cm[0][1]
        merged[1][0] += cm[1][0]
        merged[1][1] += cm[1][1]

    return merged

def benchmark_and_log(rt, config, ds, clf):
    """Slices the test set and generates the final confusion matrix"""
    def run():
        test_subsets = ds.test_data()
        params = BenchmarkParams(rt, ds, config, clf)
        cms = Parallel(n_jobs=config.j, verbose=1)(
                delayed(_benchmark)(params, subset) for subset in test_subsets)
        return _merge_confusion_matrices(cms)

    benchmark_confusion_matrix = run()
    log = logger_factory.logger_for(config, 'benchmark')
    log_utils.log_confusion_matrix(log, benchmark_confusion_matrix)
