# Flu Neural

Neural network research & testing for flu prediction, in this iteration is for binary classification, specifically.

The flu model:

- Input: 7 features (fever, cough, etc.)
- Output: Probability 0.0 → 1.0, thresholded to 0 or 1
- Task: Binary classification (Has Flu vs No Flu)

## Setup

```bash
poetry install
```

## Usage

```bash
# Generate patient data
poetry run python -m flu_neural.generate_data

# Train model
poetry run python -m flu_neural.train_model
```
