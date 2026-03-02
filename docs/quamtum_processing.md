# Quantum Processing Stage Summary

## Scope of this stage
This stage delivered two main capabilities:

1. Dataset loading from both local storage and Amazon S3.
2. Runtime model selection between a classical baseline and a quantum-hybrid model.

## What was implemented

### 1) Data loading now supports S3
Updated `src/data/loader.py` so `load_data(...)` can read datasets from:

- Local directories (existing behavior).
- S3 prefixes passed as `s3://bucket/prefix` (new behavior).

Key details:

- The loader appends the target filename (for example `train.xlsx`) to the S3 prefix.
- Reads `.xlsx`, `.xls`, and `.csv` files from in-memory bytes.
- Uses `boto3` when available.
- Includes fallback paths for public objects (unsigned/public URL fetch).
- Keeps date parsing behavior (`Date` column) unchanged.
- Raises clear `FileNotFoundError` for missing keys/objects.

### 2) Hybrid model support in training entrypoint
Updated `run.py` to expose model selection via:

- `--model-type normal|hybrid`

Behavior:

- `normal`: trains the existing `MLP`.
- `hybrid`: trains a new MerLin-based hybrid regressor.

Also added quantum-specific CLI parameters:

- `--n-modes`
- `--n-photons`
- `--quantum-depth`
- `--encoding-type` (`angle` or `amplitude`)
- `--measurement`

Checkpoint metadata now stores both classical and quantum run settings.

### 3) New hybrid model module
Added `src/hybrid/model.py` with `MerlinHybridRegressor`.

Architecture:

1. Classical projector (`Linear + Tanh`) to quantum input space.
2. Quantum encoder (`Quantificator`) from `src/quantum/quantificator.py`.
3. Classical readout head for scalar regression output.

Encoding handling:

- `angle`: projects to `n_modes` and feeds real values to the quantum encoder.
- `amplitude`: builds normalized complex amplitudes from projected real/imag parts.

## Documentation updates
Updated `README.md` with:

- New `--model-type` explanation.
- Quantum CLI argument table entries.
- Hybrid command example.
- Existing S3 usage examples retained.

## How to run

### Local data + classical model
```bash
python run.py --model-type normal
```

### Local data + hybrid model
```bash
python run.py --model-type hybrid --n-modes 4 --n-photons 2 --quantum-depth 2 --encoding-type angle --measurement probs
```

### S3 data + hybrid model
```bash
python run.py --data-dir s3://raw-721094557902-us-east-1 --model-type hybrid --n-modes 4 --n-photons 2 --quantum-depth 2 --encoding-type angle --measurement probs
```

## Dependencies and runtime notes

- `openpyxl` is required for `.xlsx` inputs.
- `torch` is required for training.
- `boto3` is recommended for private S3 buckets.
- Public S3 objects can be accessed through fallback paths without `boto3` in some environments.

## Files changed in this stage

- `src/data/loader.py`
- `run.py`
- `src/hybrid/model.py` (new)
- `README.md`
- `quamtum_processing.md` (new)
