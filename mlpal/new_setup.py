#!/bin/env python
# coding=utf-8

template = """#!/bin/env python
# coding=utf-8

from mlpal.base_datasource import BaseDataSource
from mlpal.base_learning_spec import BaseLearningSpec, LearningSetupIsBroken


class LearningSpec(BaseLearningSpec):

    def training_classifier(self):
        raise LearningSetupIsBroken('#training_classifier not implemented')


class DataSource(BaseDataSource):

    def __init__(self, config):
        pass

    def Xy(self, raw_data):
        # returns X,y
        raise LearningSetupIsBroken('#Xy not implemented')

    def raw_train(self):
        # return data that Xy understands
        raise LearningSetupIsBroken('#raw_train not implemented')

    def raw_test(self):
        # return a list of data that Xy understands
        # see BaseDataSource#raw_test docs
        raise LearningSetupIsBroken('#raw_test not implemented')
"""

def generate(setup_id):
    with open('%s.py' % setup_id, 'w') as f:
        f.write(template)
