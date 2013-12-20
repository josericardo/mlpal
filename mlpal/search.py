#!/usr/bin/env python

from train import Trainer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import logger_factory

class Searcher(Trainer):
    def __init__(self, config, data_source):
        self.log = logger_factory.logger_for(config, self.__class__.__name__)
        self.config = config
        self.ds = data_source

    def fit(self, pipelines=None):
        """ Returns the best parameterization for each pipeline """
        best = {}

        for k, p in pipelines.iteritems():
            best[k] = self.fit_pipeline(p['pipeline'], p['params'])
            self.log.info("Params for best %s: %s" % (k, best[k]['clf'].best_params_))
            self.log.info("Best score for %s: %s" % (k, best[k]['clf'].best_score_))

        return best

    def fit_pipeline(self, pipeline, parameters):
        X_train, y_train = self.ds.train_data()
        cv = StratifiedShuffleSplit(y_train, n_iter=self.config.cv, random_state=self.config.random_state)
        grid = GridSearchCV(pipeline, parameters, n_jobs=self.config.j, verbose=1, cv=cv)

        grid.fit(X_train, y_train)
        dump_path = self.save(grid)

        self.log.info("Best classifier found via GridSearch. On training data:")
        self.classify_and_report(grid.best_estimator_, X_train, y_train)

        self.log.info("Computing cv score (%d folds)..." % self.config.cv)

        # TODO grid.best_estimator_ has already been cross-validated
        # reuse the scores already computed
        scores = self._cv_scores(grid.best_estimator_, X_train, y_train)
        return {'clf': grid, 'dump_path': dump_path}
