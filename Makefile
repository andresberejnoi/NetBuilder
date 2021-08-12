init:
	pip install -r requirements.txt
	python setup.py install

test:
	py.test tests

.PHONY: init
