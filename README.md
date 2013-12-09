[![Build Status](https://travis-ci.org/gendoc/mlpal.png)](https://travis-ci.org/gendoc/mlpal)

# Machine Learning Pal

A lightweight toolkit to help you on common Machine Learning tasks.
Implement two interfaces and start experimenting.

*Warning*: this project is still in a very alpha stage.

## Usage

### Define a learning setup

For mlpal to work, you must define set module, that will contain two classes:

- DataSource
- LearningSpec

A working example of setup file is the `tests/dummy_setup.py`.

The tasks that are already supported:

* train
* search
* benchmark
* learning_curves
* plot_pca
* misclassified

## Wishlist

- [ ] Add some reasonable documentation to readthedocs
- [ ] Add required dependencies to the packaged version
- [ ] Make the build process run on travis-ci
- [ ] Use coveralls
- [ ] Improve test suite (define a desired coverage)
- [ ] Add default classifiers that can be selected via command line
- [ ] Enable the definition of params via CLI and config.json
- [ ] One configuration per project/module
- [ ] One output dir per project/module
- [ ] Most things work only for the binary classification case.
- [ ] Isolated tests (today they read the config.json file)

## Similar projects

You may also want to take a look at these projects:

* SKLL: http://skll.readthedocs.org
* nltk-trainer: http://nltk-trainer.readthedocs.org
