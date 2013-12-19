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

def _benchmark(params, test_slice):
    """Benchmarks the classifier based on a test set slice"""
    try:
        (start, end) = test_slice
        X_test, y_test = params.ds.testing_slice(start, end)
        clf = copy.deepcopy(params.clf)
        trainer = Trainer(params.rt, params.config, params.ds, params.clf)
        return trainer.benchmark(X_test, y_test).confusion_matrix
    except Exception as e:
        print(e)
        traceback.print_exc()

def _slice(length, wanted_parts=1):
    return [(i*length // wanted_parts, (i+1)*length // wanted_parts)
            for i in range(wanted_parts)]

def _merge_confusion_matrices(cms):
    merged = [[0,0], [0,0]]

    for cm in cms:
        merged[0][0] += cm[0][0]
        merged[0][1] += cm[0][1]
        merged[1][0] += cm[1][0]
        merged[1][1] += cm[1][1]

    return merged

def benchmark_and_log(rt, config, ds, clf):
    """Slices the test set and generates the final confusion matrix"""
    def run():
        test_size = ds.get_test_size()
        num_of_chunks = int(math.ceil(test_size/70000.0)) # is 70k good enough?
        test_subsets = _slice(test_size, wanted_parts=num_of_chunks)

        print("Benchmarking classifier over %d examples in %d chunks." % (test_size, num_of_chunks))

        params = BenchmarkParams(rt, ds, config, clf)
        cms = Parallel(n_jobs=config.j, verbose=1)(
                delayed(_benchmark)(params, i) for i in test_subsets)

        return _merge_confusion_matrices(cms)

    benchmark_confusion_matrix = run()
    log = logger_factory.logger_for(config, 'Benchmarker')
    log_utils.log_confusion_matrix(log, benchmark_confusion_matrix)
