# Makefile for commune.core.tx module

.PHONY: install test clean

install:
	pip install -e .

test:
	python -c "import commune as c; c.module('tx').test()"

clean:
	rm -rf __pycache__ *.pyc .pytest_cache
