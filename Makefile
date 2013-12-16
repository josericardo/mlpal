default: pylint_errors test

test:
	 nosetests -s -v $(TEST)
	 rm *.scores *.png *.log

# local update
lup: uninstall install 

uninstall:
	pip uninstall -y -q MLPal

install:
	python setup.py sdist
	pip install dist/MLPal-0.1.0.tar.gz
	rm -rf MANIFEST dist

source_up:
	git pull origin master

test-coverage:
	rm -rf .coverage coverage
	nosetests -s -v --with-coverage --cover-package=mlpal

up: source_up lup

todos:
	grep -r --include="*.py" "TODO" .

pylint_errors:
	 pylint -E --rcfile=.pylintrc mlpal/ --disable=E0611

pylint:
	 pylint --rcfile=.pylintrc mlpal/ --disable=E0611

clean:
	rm -rf htmlcov

