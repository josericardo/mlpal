#!/usr/bin/env python
# coding=utf-8

import math
import numpy as np
import pylab as pl
from scipy.stats import sem
from sklearn.cross_validation import ShuffleSplit
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

def gen_output(config, out):
    out_prefix = 'learning_curves'

    if config.o:
        out_prefix = config.o

    out_png = '%s_lcs.png' % out_prefix
    pl.savefig(out_png)
    print(out)
    with open("%s_lcs.scores" % out_prefix, "w") as f:
        f.write(out)

def train_sizes_from(n_samples, config):
    gen_space = np.logspace
    begin = math.log10(config.begin)
    end = math.log10(n_samples)

    if config.space == 'linear':
        gen_space = np.linspace
        begin = config.begin
        end = n_samples

    return gen_space(begin, end, config.points).astype(np.int)

def fit(spec, X, y, train, test, i, j):
    clf = spec.training_classifier()
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    clf.fit(X_train, y_train)

    # TODO read config.score
    return f1_score(y_train, clf.predict(X_train)), f1_score(y_test, clf.predict(X_test))

def plot_lcs(spec, X, y, config, X_test=None, y_test=None):
    train_sizes = train_sizes_from(len(X), config)
    train_scores = np.zeros((train_sizes.shape[0], config.cv), dtype=np.float)
    test_scores = np.zeros((train_sizes.shape[0], config.cv), dtype=np.float)
    clfs = {}

    len_of_max_train_size_number = len(str(train_sizes[-1]))

    out = ''
    for i, train_size in enumerate(train_sizes):
        cv = ShuffleSplit(train_size, n_iter=config.cv)

        i_scores = Parallel(n_jobs=config.j, pre_dispatch=1, verbose=1)(
                delayed(fit)
                (spec, X, y, train, test, i, j)
                for j, (train, test) in enumerate(cv)
        )

        for j, score in enumerate(i_scores):
            train_scores[i,j] = score[0]
            test_scores[i,j] = score[1]

        test_mean = np.mean(test_scores[i])
        train_mean = np.mean(train_scores[i])

        out += "For %s examples: " % str(train_size).rjust(len_of_max_train_size_number, ' ')
        out += "mean train score=%.2f ; mean cv score=%.2f\n" % (train_mean, test_mean)

    mean_train = np.mean(train_scores, axis=1)
    confidence = sem(train_scores, axis=1) * 2

    pl.fill_between(train_sizes, mean_train - confidence, mean_train + confidence,
                    color = 'b', alpha = .2)
    pl.plot(train_sizes, mean_train, 'o-k', c='b', label='Train score')

    mean_test = np.mean(test_scores, axis=1)
    confidence = sem(test_scores, axis=1) * 2

    pl.fill_between(train_sizes, mean_test - confidence, mean_test + confidence,
                    color = 'g', alpha = .2)
    pl.plot(train_sizes, mean_test, 'o-k', c='g', label='Test score')

    pl.xlabel('Training set size')
    pl.ylabel('Score')
    #pl.xlim(0, X.shape[0])
    pl.ylim((None, 1.01))  # The best possible score is 1.0
    pl.legend(loc='best')
    pl.title('Main train and test scores +/- 2 standard errors')

    gen_output(config, out)
    return clfs
