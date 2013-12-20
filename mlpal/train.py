#!/usr/bin/env python

import os
import joblib
from collections import namedtuple
import numpy as np
from datetime import datetime
from sklearn import metrics

from .log_utils import log_confusion_matrix
from . import cross_validation
from . import logger_factory


class Trainer:
    def __init__(self, rt, config, data_source, classifier):
        """
        Params
        ------
        rt: MLPalRuntime (see runner#run)
        config: args
        data_source: a BaseDataSource
        classifier: a scikit-learn classifier
        """
        self.rt = rt
        self.log = logger_factory.logger_for(config, self.__class__.__name__)
        self.classifier = classifier
        self.ds = data_source
        self.config = config

    def run(self):
        self.log.info("Extracting features and labels for training.")
        X_train, y_train = self.ds.train_data()
        self.log.info("Data loaded, fitting classifier.")

        self.classifier.fit(X_train, y_train)
        self.log.info("Classifier trained.")
        self.save(self.classifier)

        self.log.info("Evaluating classifier on the training data.")
        train_score, confusion_matrix = self.classify_and_report(self.classifier, X_train, y_train)

        self.log.info("Computing cv score (%d folds)..." % self.config.cv)
        scores = self._cv_scores(self.classifier, X_train, y_train)
        self.add_training_info_to_history(train_score, scores)

    def _cv_scores(self, clf, X, y):
        scores =  cross_validation.cross_validate(clf, X, y, self.config)

        self.log.info("Cross-Validation %s score: %0.2f (+/- %0.2f)"
            % (self.config.scoring, scores.mean(), scores.std() * 2))

        return scores

    def add_training_info_to_history(self, train_score, scores):
        e = self.rt.info
        e['clf'] = self.classifier.__repr__()
        e['cv_scores_mean'] = scores.mean()
        e['cv_scores_std'] = scores.std()
        e['train_f1_score'] = train_score

    def benchmark(self, X_test, y_test):
        return self.classify_and_report(self.classifier, X_test, y_test, print_report=False)

    def classify_and_report(self, clf, X, y, print_report=True):
        """returns (f1-score, confusion_matrix)"""
        y_predicted = clf.predict(X)

        # TODO score according to config.scoring
        f1 = metrics.f1_score(y, y_predicted)
        cm = self._get_metrics(y_predicted, y, print_report=print_report)
        return namedtuple('TrainingResults', ['score', 'confusion_matrix'])(f1, cm)

    def _get_metrics(self, y_predicted, y, print_report):
        self.log.info("Generating metrics")

        if print_report:
            report = metrics.classification_report(y, y_predicted)
            self.log.info("\n" + report)

        cm = metrics.confusion_matrix(y, y_predicted)
        log_confusion_matrix(self.log, cm)
        return cm

    def save(self, clf):
        if not os.path.isdir('dumps'):
            os.makedirs('dumps')

        dump = "dumps/last.dump"
        self.log.info("Saving classifier to %s..." % dump)
        joblib.dump(clf, dump, compress=1)
        return dump
