Master | Master | Develop | Develop
-------|--------|---------|--------
[![Build Status](https://travis-ci.org/gendoc/mlpal.png?branch=master)](https://travis-ci.org/gendoc/mlpal) | [![Coverage Status](https://coveralls.io/repos/josericardo/mlpal/badge.png?branch=master)](https://coveralls.io/r/josericardo/mlpal?branch=master) | [![Build Status](https://travis-ci.org/gendoc/mlpal.png?branch=master)](https://travis-ci.org/gendoc/mlpal) | [![Coverage Status](https://coveralls.io/repos/josericardo/mlpal/badge.png?branch=develop)](https://coveralls.io/r/josericardo/mlpal?branch=develop)


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

`$ mlpal -h`

`$ mlpal task -h`

#### Uninstalling

`$ make uninstall`

### Define a learning setup

For mlpal to work, you must define a setup module, that will contain two classes:

- `DataSource`
- `LearningSpec`

You can generate a new setup via:

`$ mlpal new_setup my_project`

and fill the blanks.

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
