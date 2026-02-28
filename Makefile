.PHONY: venv install fmt lint baseline hybrid all clean

PY := .venv/bin/python
PIP := .venv/bin/pip

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	$(PIP) torch torchvision --index-url https://download.pytorch.org/whl/cu126

fmt:
	$(PY) -m ruff format .

lint:
	$(PY) -m ruff check .

baseline:
	$(PY) run.py --config configs/baseline.yaml

hybrid:
	$(PY) run.py --config configs/hybrid.yaml

all: baseline hybrid

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__
	rm -rf results/*
