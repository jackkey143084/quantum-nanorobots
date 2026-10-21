.PHONY: install install-all test demo serve lint clean

install:
	pip install -e .

install-all:
	pip install -e ".[all,dev]"

test:
	python -m pytest -q

demo:
	PYTHONPATH=src python -m qnrs.cli demo

serve:
	PYTHONPATH=src python -m qnrs.cli serve --host 0.0.0.0 --port 8080

examples:
	PYTHONPATH=src python examples/run_examples.py

lint:
	python -m compileall -q src tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
