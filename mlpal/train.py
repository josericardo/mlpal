#!/usr/bin/env python

import os
import joblib
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation

from log_utils import log_confusion_matrix
import logger_factory

class Trainer:
    def __init__(self, config, data_source, classifier):
        """
        Params
        ------
        config: args
        data_source: a BaseDataSource
        classifier: a scikit-learn classifier
        """
        self.log = logger_factory.logger_for(config, self.__class__.__name__)
        self.classifier = classifier
        self.ds = data_source
        self.config = config

    def run(self):
        self.log.info("Extracting features and labels for training")
        X_train, y_train = self.ds.train_data()

        self.classifier.fit(X_train, y_train)
        self.save(self.classifier)

        self.log.info("Training evaluation")
        self.classify_and_report(self.classifier, X_train, y_train)

        scores = cross_validation.cross_val_score(self.classifier, X_train, y_train, cv=self.config.cv, scoring='f1')

        self.log.info("Cross-Validation score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        #self.log.info("= Validation reports =")
        #X_validation, y_validation = self.ds.validation_data()

    def benchmark(self, X_test, y_test):
        return self.classify_and_report(self.classifier, X_test, y_test, print_report=False)

    def classify_and_report(self, clf, X_test, y_test, print_report=True):
        y_predicted = clf.predict(X_test)
        return self.get_metrics(y_predicted, y_test, print_report=print_report)

    def get_metrics(self, y_predicted, y, print_report):
        self.log.info("Generating metrics")

        if print_report:
            report = classification_report(y, y_predicted)
            self.log.info("\n" + report)

        # C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j:\n")
        cm = confusion_matrix(y, y_predicted)
        log_confusion_matrix(self.log, cm)
        return cm

    def save(self, clf):
        if not os.path.isdir('dumps'):
            os.makedirs('dumps')

        dump = "dumps/last.dump"
        self.log.info("Saving classifier to %s..." % dump)
        joblib.dump(clf, dump, compress=1)
        return dump
