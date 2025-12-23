"""
================================================================================
FLU_NEURAL - A Neural Network Tutorial Package
================================================================================

This package demonstrates how to build and train a neural network for
flu prediction using PyTorch.

MODULES:
    generate_data.py  - Creates synthetic patient data, saves to CSV
    train_model.py    - Loads CSV, trains neural network, evaluates

USAGE:
    Step 1: Generate training data
        poetry run python -m flu_neural.generate_data

    Step 2: Train the model
        poetry run python -m flu_neural.train_model

    Or run both in sequence:
        poetry run python -m flu_neural.main

WORKFLOW DIAGRAM:
    +------------------+       +------------------+       +------------------+
    | generate_data.py | ----> | data/patients.csv| ----> | train_model.py   |
    +------------------+       +------------------+       +------------------+
    | Creates patients |       | YOU CAN EDIT!    |       | Reads CSV        |
    | with symptoms    |       | Tamper with data |       | Trains network   |
    | Saves to CSV     |       | Add edge cases   |       | Shows results    |
    +------------------+       +------------------+       +------------------+
"""

from flu_neural.generate_data import main as generate_main
from flu_neural.train_model import main as train_main


def main():
    """Run the complete pipeline: generate data then train model."""

    print("=" * 65)
    print("FLU NEURAL - COMPLETE PIPELINE")
    print("=" * 65)
    print("\nThis will run both steps:")
    print("  1. Generate patient data -> data/patients.csv")
    print("  2. Train neural network on that data")
    print("\n" + "=" * 65)

    # Step 1: Generate data
    print("\n>>> STEP 1: GENERATING DATA <<<\n")
    generate_main()

    # Step 2: Train model
    print("\n\n>>> STEP 2: TRAINING MODEL <<<\n")
    train_main()


if __name__ == "__main__":
    main()
