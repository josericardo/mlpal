from distutils.core import setup

from mlpal.version import __version__

setup(
    name='MLPal',
    version=__version__,
    author='Gendoc',
    author_email='josericardo@gendoc.com.br',
    packages=['mlpal', 'mlpal.tests'],
    scripts=['bin/mlpal'],
    url='http://pypi.python.org/pypi/mlpal/',
    license='LICENSE.txt',
    description='Lightweight framework to help on common Machine Learning tasks.',
    long_description=open('README.md').read(),
    install_requires=[
        #"Django >= 1.1.1",
        #"caldav == 0.1.4",
    ],
)
