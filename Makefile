PYTHON ?= python
REPORT ?= tests/assets/sample_report.json

.PHONY: setup test regression demo-help

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

regression:
	$(PYTHON) scripts/run_regression_suite.py --report $(REPORT)

demo-help:
	$(PYTHON) scripts/demo_friend.py --help

