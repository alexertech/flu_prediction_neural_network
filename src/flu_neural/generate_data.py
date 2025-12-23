"""
================================================================================
GENERATE_DATA.PY - Patient Data Generator
================================================================================

This script generates synthetic patient data for flu prediction training.

WHAT IT DOES:
    1. Creates N synthetic patients with realistic symptoms
    2. Assigns flu diagnosis based on symptom patterns
    3. Saves everything to a CSV file for later use

OUTPUT FILE: ../data/patients.csv (relative to project root)

USAGE:
    poetry run python -m flu_neural.generate_data
    poetry run python -m flu_neural.generate_data --samples 5000
    poetry run python -m flu_neural.generate_data --output my_patients.csv

THE DATA SCHEMA:
+------------------------------------------------------------------+
| COLUMN           | TYPE    | RANGE          | DESCRIPTION        |
+------------------+---------+----------------+--------------------+
| fever            | float   | 0.0 - 1.0      | 0=none, 1=high     |
| cough            | int     | 0 or 1         | presence of cough  |
| sore_throat      | int     | 0 or 1         | presence           |
| body_ache        | int     | 0 or 1         | muscle pain        |
| fatigue          | float   | 0.0 - 1.0      | 0=energetic, 1=exhausted |
| age_normalized   | float   | 0.05 - 0.85    | age/100            |
| flu_contact      | int     | 0 or 1         | exposed to flu?    |
| has_flu          | int     | 0 or 1         | TARGET LABEL       |
+------------------------------------------------------------------+

HOW FLU IS DETERMINED (the "hidden rules" the model must discover):
    flu_score = fever * 3.0      (strong indicator)
              + cough * 1.5      (moderate)
              + sore_throat * 0.5 (weak)
              + body_ache * 1.0  (moderate)
              + fatigue * 1.0    (moderate)
              + flu_contact * 2.0 (strong)
              + elderly * 0.5    (slight boost if age > 60)

    + random noise to simulate real-world unpredictability
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_patient_data(n_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic patient data for flu prediction.

    Args:
        n_samples: Number of patients to generate
        random_seed: Seed for reproducibility (same seed = same data)

    Returns:
        DataFrame with patient features and flu diagnosis
    """
    # Set seed for reproducibility
    np.random.seed(random_seed)

    # Initialize storage for all patients
    patients = []

    for i in range(n_samples):
        # =====================================================================
        # GENERATE RANDOM PATIENT FEATURES
        # =====================================================================
        fever = np.random.uniform(0, 1)           # 0 = no fever, 1 = high fever
        cough = np.random.choice([0, 1])          # binary: has cough or not
        sore_throat = np.random.choice([0, 1])    # binary
        body_ache = np.random.choice([0, 1])      # binary
        fatigue = np.random.uniform(0, 1)         # 0 = energetic, 1 = exhausted
        age_normalized = np.random.uniform(0.05, 0.85)  # age/100 (5-85 years)
        flu_contact = np.random.choice([0, 1])    # was exposed to flu patient?

        # =====================================================================
        # DETERMINE FLU DIAGNOSIS (the hidden rules)
        # =====================================================================
        # This is what the neural network must DISCOVER from the data!
        # We're essentially encoding medical knowledge as weights.

        flu_score = (
            fever * 3.0 +           # Fever is the strongest indicator
            cough * 1.5 +           # Cough is moderate
            sore_throat * 0.5 +     # Sore throat is weak (many causes)
            body_ache * 1.0 +       # Body ache is moderate
            fatigue * 1.0 +         # Fatigue is moderate
            flu_contact * 2.0 +     # Contact is very important
            (age_normalized > 0.6) * 0.5  # Elderly slightly more susceptible
        )

        # Convert to probability (normalize to 0-1 range)
        flu_probability = flu_score / 8.0
        flu_probability = np.clip(flu_probability, 0, 1)

        # Add noise (real medicine isn't perfectly predictable)
        noise = np.random.uniform(-0.15, 0.15)
        flu_probability = np.clip(flu_probability + noise, 0, 1)

        # Final diagnosis: threshold at 0.5
        has_flu = 1 if flu_probability > 0.5 else 0

        # Store this patient
        patients.append({
            'fever': round(fever, 4),
            'cough': cough,
            'sore_throat': sore_throat,
            'body_ache': body_ache,
            'fatigue': round(fatigue, 4),
            'age_normalized': round(age_normalized, 4),
            'flu_contact': flu_contact,
            'has_flu': has_flu
        })

    # Convert to DataFrame
    df = pd.DataFrame(patients)
    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """Print a summary of the generated dataset."""

    print("=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)

    print(f"\nTotal patients generated: {len(df)}")
    print(f"Patients WITH flu:        {df['has_flu'].sum()} ({df['has_flu'].mean()*100:.1f}%)")
    print(f"Patients WITHOUT flu:     {len(df) - df['has_flu'].sum()} ({(1-df['has_flu'].mean())*100:.1f}%)")

    print("\n" + "-" * 65)
    print("SAMPLE PATIENTS (first 10):")
    print("-" * 65)
    print(f"{'Fever':>7} {'Cough':>6} {'Throat':>7} {'Ache':>5} {'Fatigue':>8} {'Age':>6} {'Contact':>8} | {'FLU':>4}")
    print("-" * 65)

    for i, row in df.head(10).iterrows():
        print(f"{row['fever']:>7.2f} {row['cough']:>6} {row['sore_throat']:>7} "
              f"{row['body_ache']:>5} {row['fatigue']:>8.2f} {row['age_normalized']:>6.2f} "
              f"{row['flu_contact']:>8} | {row['has_flu']:>4}")

    print("\n" + "-" * 65)
    print("FEATURE STATISTICS:")
    print("-" * 65)
    print(df.describe().round(3).to_string())


def main():
    # =========================================================================
    # PARSE COMMAND LINE ARGUMENTS
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='Generate synthetic patient data for flu prediction training'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=1000,
        help='Number of patients to generate (default: 1000)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output CSV file path (default: data/patients.csv in project root)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # =========================================================================
    # DETERMINE OUTPUT PATH
    # =========================================================================
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: create data/ folder in project root
        # Navigate from src/flu_neural/ up to flu_neural/
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / "patients.csv"

    # =========================================================================
    # GENERATE DATA
    # =========================================================================
    print("\n" + "=" * 65)
    print("PATIENT DATA GENERATOR")
    print("=" * 65)
    print(f"\nGenerating {args.samples} patients with seed={args.seed}...")

    df = generate_patient_data(n_samples=args.samples, random_seed=args.seed)

    # =========================================================================
    # DISPLAY SUMMARY
    # =========================================================================
    print_data_summary(df)

    # =========================================================================
    # SAVE TO CSV
    # =========================================================================
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("FILE SAVED")
    print("=" * 65)
    print(f"\nOutput file: {output_path.absolute()}")
    print(f"File size:   {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\nYou can now:")
    print(f"  1. Open and edit the CSV manually to experiment")
    print(f"  2. Run train_model.py to train on this data")
    print(f"\nCommand to train:")
    print(f"  poetry run python -m flu_neural.train_model")


if __name__ == "__main__":
    main()
