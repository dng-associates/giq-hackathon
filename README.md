# QIG Hackathon 2026

Repository for a hybrid classical + quantum option-pricing workflow.

## Pipeline

The project is organized around the following development pipeline.

```mermaid
flowchart LR
    subgraph M["Modeling Dev"]
        P["Preprocessing"]
        B["Baseline Models"]
        MT["Metrics"]
    end

    subgraph Q["Quantum Dev"]
        QC["Quantum Circuit"]
        FE["Feature Encoder"]
        QC --> FE
    end

    subgraph PL["Platform Dev"]
        PI["Pipeline Integration"]
        ER["Experiment Runner"]
        V["Visualization"]
        PI --> ER --> V
    end

    P --> PI
    B --> PI
    FE --> PI
```

## Stage Responsibilities

### Modeling Dev
- `Preprocessing`: load market data, clean it, build features, and prepare train/validation/test splits.
- `Baseline Models`: implement classical references to compare against the hybrid approach.
- `Metrics`: define shared evaluation metrics for all experiments (for example MAE, RMSE, R2).

### Quantum Dev
- `Quantum Circuit`: design and validate the quantum circuit topology.
- `Feature Encoder`: encode classical inputs into quantum-ready features and expose them in a reusable interface.

### Platform Dev
- `Pipeline Integration`: join classical preprocessing, quantum features, and model training into one runnable flow.
- `Experiment Runner`: execute reproducible runs using config files and track outputs.
- `Visualization`: generate result plots and comparison dashboards.

## Repository Layout

Current scaffold:

```text
qig-hackathon/
|-- DATASETS/
|   |-- train.xlsx
|   |-- test_template.xlsx
|   `-- sample_Simulated_Swaption_Price.xlsx
|-- configs/
|   |-- baseline.yaml
|   `-- hybrid.yaml
|-- src/
|   |-- classical/
|   |-- data/
|   |-- eval/
|   |-- hybrid/
|   `-- quantum/
|-- run.py
|-- Makefile
`-- requirements.txt
```

Suggested mapping to the pipeline:

- `src/data` -> `Preprocessing`
- `src/classical` -> `Baseline Models`
- `src/eval` -> `Metrics` and `Visualization`
- `src/quantum` -> `Quantum Circuit` and `Feature Encoder`
- `src/hybrid` -> `Pipeline Integration` and `Experiment Runner`

## How To Run

Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run experiments:

```bash
python run.py --config configs/baseline.yaml
python run.py --config configs/hybrid.yaml
```

Or with Make:

```bash
make baseline
make hybrid
```

## Implementation Checklist

- [ ] Preprocessing pipeline in `src/data`
- [ ] Baseline models in `src/classical`
- [ ] Quantum circuit + encoder in `src/quantum`
- [ ] Integrated training pipeline in `src/hybrid`
- [ ] Metrics + visual reports in `src/eval`
- [ ] Config-driven runs from `run.py`
