.PHONY: venv install fmt lint baseline hybrid all clean data-raw data-refined data clean-data data-sync data-list

DATASET_BUCKET ?= dataset-721094557902-us-east-1
AWS_REGION ?= us-east-1
DATA_RAW_DIR := data/raw
DATA_REFINED_DIR := data/refined

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
	$(PY) run.py --model-type normal

hybrid:
	$(PY) run.py --model-type hybrid --quantum-backend merlin

all: baseline hybrid

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__
	rm -rf results/*
	rm -rf $(DATA_RAW_DIR)/*

data-raw:
	@mkdir -p $(DATA_RAW_DIR)
	aws --region $(AWS_REGION) s3 sync s3://$(DATASET_BUCKET)/raw/v1 $(DATA_RAW_DIR) \
		--no-sign-request --only-show-errors --size-only

data-refined:
	@mkdir -p $(DATA_REFINED_DIR)
	aws --region $(AWS_REGION) s3 sync s3://$(DATASET_BUCKET)/refined/v1 $(DATA_REFINED_DIR) \
		--no-sign-request --only-show-errors --size-only

data: data-raw data-refined

clean-data:
	rm -rf $(DATA_RAW_DIR) $(DATA_REFINED_DIR)

data-sync:
	@mkdir -p $(DATA_RAW_DIR)
	aws --region $(AWS_REGION) s3 cp s3://$(DATASET_BUCKET)/raw/sample_Simulated_Swaption_Price.xlsx $(DATA_RAW_DIR)/ --no-sign-request --only-show-errors
	aws --region $(AWS_REGION) s3 cp s3://$(DATASET_BUCKET)/raw/test_template.xlsx $(DATA_RAW_DIR)/ --no-sign-request --only-show-errors
	aws --region $(AWS_REGION) s3 cp s3://$(DATASET_BUCKET)/raw/train.xlsx $(DATA_RAW_DIR)/ --no-sign-request --only-show-errors

data-list:
	aws --region $(AWS_REGION) s3 ls s3://$(DATASET_BUCKET)/

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

set-dataset:
	./scripts/s3_upload_raw.sh
