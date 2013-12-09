#!/usr/bin/env python

import copy
import traceback
import math
from joblib import Parallel, delayed
from train import Trainer
from log_utils import log_confusion_matrix

def slices(length, wanted_parts=1):
    return [(i*length // wanted_parts, (i+1)*length // wanted_parts)
            for i in range(wanted_parts)]

def benchmark(benchmarker, test_slice):
    try:
        (start, end) = test_slice
        X_test, y_test = benchmarker.ds.testing_slice(start, end)
        clf = copy.deepcopy(benchmarker.clf)
        trainer = Trainer(benchmarker.config, benchmarker.ds, clf)
        return trainer.benchmark(X_test, y_test)
    except Exception as e:
        print(e)
        traceback.print_exc()

class Benchmarker:
    def __init__(self, config, ds, classifier):
        self.ds = ds
        self.clf = classifier
        self.log = config.logger_for(self.__class__.__name__)
        self.config = config

    def run(self):
        test_size = self.ds.get_test_size()
        num_of_chunks = int(math.ceil(test_size/70000.0)) # 70k is good enough
        test_subsets = slices(test_size, wanted_parts=num_of_chunks)

        print("Benchmarking classifier over %d examples in %d chunks." % (test_size, num_of_chunks))

        cms = Parallel(n_jobs=self.config.n_jobs, verbose=1)(delayed(benchmark)(self, i) for i in test_subsets)

        final_confusion_matrix = [[0,0], [0,0]]

        for cm in cms:
            final_confusion_matrix[0][0] += cm[0][0]
            final_confusion_matrix[0][1] += cm[0][1]
            final_confusion_matrix[1][0] += cm[1][0]
            final_confusion_matrix[1][1] += cm[1][1]

        log_confusion_matrix(self.log, final_confusion_matrix)
