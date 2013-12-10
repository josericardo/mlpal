#!/usr/bin/env python

import pickle
import os
import numpy as np

from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation

from log_utils import log_confusion_matrix

class Trainer:
    def __init__(self, config, logs_factory, data_source, classifier):
        """
        Params
        ------
        config: args
        logs_factory: implements #logger_for
        data_source: a BaseDataSource
        classifier: a scikit-learn classifier
        """
        self.log = logs_factory.logger_for(self.__class__.__name__)
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
        now = datetime.now().strftime("%Y%m%dat%H%M%S")
        return self.dump('clf', now, clf)

    def dump(self, kind, sufix, obj):
        if not os.path.isdir('dumps'):
            os.makedirs('dumps')

        dump = "%s%s.pickle" % (kind, sufix)

        self.log.info("Saving %s to %s..." % (kind, dump))
        with open("dumps/%s" % dump, 'wb') as f:
            pickle.dump(obj, f)

        self.log.info("Linking as the last dump")
        if os.path.isfile("dumps/last.pickle"):
            os.unlink('dumps/last.pickle')
        os.symlink(dump, 'dumps/last.pickle')
        return dump
