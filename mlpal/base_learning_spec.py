#!/usr/bin/env python

class LearningSetupIsBroken(RuntimeError):
    pass

class BaseLearningSpec:
    """ Helper to make a LearningSpec definition easier 

    There's only one method that must be overriden: #training_classifier

    Then, you can choose to define:
    
    #gridsearch_pipelines: a custom GridSearch pipeline

    #gridsearch_params: define only the params for the #training_classifier
                        and let the GridSearch pipeline be generated 
                        automatically.

    *Warning* if you don't define any of the two above methods
              the search will be looking for nothing.
    """
    def _location(self):
        return (__file__, self.__class__.__name__)

    def training_classifier(self):
        error = "%s#%s must define the #training_classifier method" % self._location()
        raise LearningSetupIsBroken(error)

    def gridsearch_params(self):
        return {'default': {}}

    def gridsearch_pipelines(self):
        if 'default' not in  self.gridsearch_params():
            error = "#gridsearch_params must return a hash with the 'default' key set"
            raise LearningSetupIsBroken(error)

        return {'default': {
            'pipeline': self.training_classifier(),
            'params': self.gridsearch_params()['default']
            }
        }
