test:
	 nosetests -s -v $(TEST)

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

up: source_up lup
