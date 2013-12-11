[![Build Status](https://travis-ci.org/gendoc/mlpal.png)](https://travis-ci.org/gendoc/mlpal)

# Machine Learning Pal

A lightweight toolkit to help you on common Machine Learning tasks.
Implement two interfaces and start experimenting.

*Warning*: this project is still in a __very alpha__ stage.

## Usage

### Running

To install the package locally:

```
$ pip install -r requirements.txt
$ make install
```

Using:

`$ mlpal --help`

#### Uninstalling

`$ make uninstall`

### Define a learning setup

For mlpal to work, you must define a setup module, that will contain two classes:

- `DataSource`
- `LearningSpec`

A working example of setup file is the `tests/dummy_setup.py`.

The tasks that are already supported:

* train
* search
* benchmark
* learning_curves
* plot_pca
* misclassified

Most command line parameters can have their default value defined in the
`config.yaml` file.

## Contributing

Random acts of kindness are more than welcome. Besides the Github issues,
known smells in the code can be listed via:

`$ make todos`

## Similar projects

You may also want to take a look at these projects:

* SKLL: http://skll.readthedocs.org
* nltk-trainer: http://nltk-trainer.readthedocs.org
