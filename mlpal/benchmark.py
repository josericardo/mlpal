#!/usr/bin/env python

import copy
from collections import namedtuple
import traceback
import math
from joblib import Parallel, delayed
from train import Trainer
from log_utils import log_confusion_matrix
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

def benchmark(rt, config, ds, clf):
    """Slices the test set and generates the final confusion matrix"""
    def run():
        test_size = ds.get_test_size()
        num_of_chunks = int(math.ceil(test_size/70000.0)) # is 70k good enough?
        test_subsets = slices(test_size, wanted_parts=num_of_chunks)

        print("Benchmarking classifier over %d examples in %d chunks." % (test_size, num_of_chunks))

        params = BenchmarkParams(rt, ds, config, clf)
        cms = Parallel(n_jobs=config.j, verbose=1)(
                delayed(_benchmark)(params, i) for i in test_subsets)
        log_final_confusion_matrix(cms)

    def slices(length, wanted_parts=1):
        return [(i*length // wanted_parts, (i+1)*length // wanted_parts)
                for i in range(wanted_parts)]

    def log_final_confusion_matrix( cms):
        final_confusion_matrix = [[0,0], [0,0]]

        for cm in cms:
            final_confusion_matrix[0][0] += cm[0][0]
            final_confusion_matrix[0][1] += cm[0][1]
            final_confusion_matrix[1][0] += cm[1][0]
            final_confusion_matrix[1][1] += cm[1][1]

        log_confusion_matrix(log, final_confusion_matrix)

    log = logger_factory.logger_for(config, 'Benchmarker')
    run()
