.PHONY: venv install fmt lint baseline hybrid all clean

PY := .venv/bin/python
PIP := .venv/bin/pip

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu126

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

.PHONY: terraform-init

TERRAFORM_DIR := ./terraform/
TERRAFORM := terraform -chdir=$(TERRAFORM_DIR)


terraform-init:
ifneq ("$(wildcard $(TERRAFORM_DIR))","")
	$(TERRAFORM) init -migrate-state
else
	$(TERRAFORM) init
endif
	
terraform-fmt:
	$(TERRAFORM) fmt -recursive

terraform-validate:
	$(TERRAFORM) validate

terraform-plan: terraform-init
	$(TERRAFORM) plan -out=$(PLAN_FILE)

terraform-apply:
	$(TERRAFORM) apply $(PLAN_FILE)

terraform-destroy:
	$(TERRAFORM) destroy

terraform-routine: terraform-plan terraform-apply

.PHONY: terraform-init terraform-fmt terraform-validate terraform-plan terraform-apply terraform-destroy terraform-routine
