"""
================================================================================
TRAIN_MODEL.PY - Neural Network Training Pipeline
================================================================================

This script reads patient data from CSV and trains a neural network to predict flu.

WHAT IT DOES:
    1. Loads patient data from CSV file
    2. Converts data to PyTorch tensors
    3. Splits into training/testing sets
    4. Builds and trains a neural network
    5. Evaluates and visualizes results

INPUT FILE: ../data/patients.csv (or specify with --input)

USAGE:
    poetry run python -m flu_neural.train_model
    poetry run python -m flu_neural.train_model --input my_data.csv
    poetry run python -m flu_neural.train_model --epochs 200 --lr 0.001

TENSOR SHAPES REFERENCE:
+------------------------------------------------------------------+
| VARIABLE        | SHAPE         | MEANING                        |
+-----------------+---------------+--------------------------------+
| X_train         | (N, 7)        | N patients, 7 features         |
| y_train         | (N, 1)        | N labels (flu: yes/no)         |
| layer1 weights  | (7, 16)       | 7 inputs  -> 16 neurons        |
| layer2 weights  | (16, 8)       | 16 neurons -> 8 neurons        |
| layer3 weights  | (8, 1)        | 8 neurons  -> 1 output         |
+------------------------------------------------------------------+

NETWORK ARCHITECTURE:
    INPUT (7 features)
        |
        v
    [Linear 7->16] -> [ReLU] -> (learns basic symptom patterns)
        |
        v
    [Linear 16->8] -> [ReLU] -> (combines patterns)
        |
        v
    [Linear 8->1] -> [Sigmoid] -> OUTPUT (flu probability 0-1)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==============================================================================
# NEURAL NETWORK DEFINITION
# ==============================================================================

class FluPredictor(nn.Module):
    """
    A feedforward neural network for flu prediction.

    Architecture:
        Input (7) -> Hidden (16) -> Hidden (8) -> Output (1)

    Layer explanation:
        - Linear: y = Wx + b (matrix multiplication + bias)
        - ReLU: max(0, x) - introduces non-linearity
        - Sigmoid: 1/(1+e^-x) - squashes to probability [0,1]
    """

    def __init__(self, input_size: int = 7, hidden1: int = 16, hidden2: int = 8):
        super(FluPredictor, self).__init__()

        # Layer definitions
        self.layer1 = nn.Linear(input_size, hidden1)  # Input -> Hidden1
        self.layer2 = nn.Linear(hidden1, hidden2)      # Hidden1 -> Hidden2
        self.layer3 = nn.Linear(hidden2, 1)            # Hidden2 -> Output

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor, shape (batch_size, 7)

        Returns:
            Output tensor, shape (batch_size, 1) - flu probabilities
        """
        # Layer 1
        x = self.layer1(x)   # (batch, 7)  -> (batch, 16)
        x = self.relu(x)

        # Layer 2
        x = self.layer2(x)   # (batch, 16) -> (batch, 8)
        x = self.relu(x)

        # Output layer
        x = self.layer3(x)   # (batch, 8)  -> (batch, 1)
        x = self.sigmoid(x)  # Probability output

        return x


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load patient data from CSV file.

    Expected columns:
        fever, cough, sore_throat, body_ache, fatigue, age_normalized, flu_contact, has_flu

    Args:
        csv_path: Path to CSV file

    Returns:
        X: Feature matrix, shape (n_samples, 7)
        y: Labels, shape (n_samples,)
    """
    print(f"\nLoading data from: {csv_path}")

    df = pd.read_csv(csv_path)

    # Feature columns (inputs to the network)
    feature_cols = ['fever', 'cough', 'sore_throat', 'body_ache',
                    'fatigue', 'age_normalized', 'flu_contact']

    # Target column (what we predict)
    target_col = 'has_flu'

    X = df[feature_cols].values
    y = df[target_col].values

    print(f"Loaded {len(X)} patients with {X.shape[1]} features")

    return X, y


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 100,
    learning_rate: float = 0.01,
    print_every: int = 20
) -> list[float]:
    """
    Train the neural network.

    Args:
        model: The neural network
        X_train: Training features tensor
        y_train: Training labels tensor
        epochs: Number of training iterations over full dataset
        learning_rate: Step size for weight updates
        print_every: Print progress every N epochs

    Returns:
        List of loss values per epoch (for plotting)
    """
    # Loss function: Binary Cross Entropy
    # Measures: how different are predictions from true labels?
    # Perfect = 0.0, Random guessing = ~0.69
    criterion = nn.BCELoss()

    # Optimizer: Adam
    # A smart gradient descent that adapts learning rate per parameter
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    print(f"\nTraining for {epochs} epochs with lr={learning_rate}...")
    print("-" * 55)

    for epoch in range(epochs):
        # ---------------------------------------------------------------------
        # FORWARD PASS: Data flows through network
        # ---------------------------------------------------------------------
        model.train()
        predictions = model(X_train)

        # ---------------------------------------------------------------------
        # CALCULATE LOSS: How wrong are we?
        # ---------------------------------------------------------------------
        loss = criterion(predictions, y_train)
        loss_history.append(loss.item())

        # ---------------------------------------------------------------------
        # BACKWARD PASS: Calculate gradients
        # ---------------------------------------------------------------------
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients

        # ---------------------------------------------------------------------
        # UPDATE WEIGHTS: Gradient descent step
        # ---------------------------------------------------------------------
        optimizer.step()

        # Print progress
        if (epoch + 1) % print_every == 0:
            with torch.no_grad():
                train_preds = (predictions > 0.5).float()
                train_acc = (train_preds == y_train).float().mean()
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Accuracy: {train_acc:.2%}")

    print("-" * 55)
    print("Training complete!")

    return loss_history


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> float:
    """
    Evaluate model on test data.

    Args:
        model: Trained neural network
        X_test: Test features tensor
        y_test: Test labels tensor

    Returns:
        Accuracy score
    """
    model.eval()

    with torch.no_grad():
        predictions = model(X_test)
        predictions_binary = (predictions > 0.5).float()

    y_test_np = y_test.numpy()
    y_pred_np = predictions_binary.numpy()

    accuracy = accuracy_score(y_test_np, y_pred_np)

    print("\n" + "=" * 55)
    print("TEST RESULTS")
    print("=" * 55)
    print(f"\nTest Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print("-" * 55)
    print(classification_report(y_test_np, y_pred_np,
                                target_names=['No Flu', 'Has Flu']))

    return accuracy


# ==============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ==============================================================================

def analyze_feature_importance(model: nn.Module) -> None:
    """
    Analyze which features the model considers most important.

    This looks at the weights in layer 1 to see which input features
    have the largest impact on the hidden layer activations.
    """
    print("\n" + "=" * 55)
    print("LEARNED FEATURE IMPORTANCE")
    print("=" * 55)

    feature_names = ['Fever', 'Cough', 'Throat', 'Ache',
                     'Fatigue', 'Age', 'Contact']

    # Get layer 1 weights
    # Shape: (16 neurons, 7 inputs) - each row is one neuron's weights
    layer1_weights = model.layer1.weight.data.numpy()

    # Average absolute weight per input feature across all neurons
    importance = np.abs(layer1_weights).mean(axis=0)

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]

    print("\nFeature importance (higher = more influential):")
    print("-" * 40)
    print(f"{'Feature':<12} | {'Importance':>10} | {'Bar':<15}")
    print("-" * 40)

    max_imp = importance.max()
    for idx in sorted_idx:
        bar_len = int(15 * importance[idx] / max_imp)
        bar = '#' * bar_len
        print(f"{feature_names[idx]:<12} | {importance[idx]:>10.4f} | {bar}")

    print("\nInterpretation:")
    print(f"  The model learned that '{feature_names[sorted_idx[0]]}' is most predictive of flu.")
    print(f"  Compare this to our data generation where fever (3.0) and contact (2.0)")
    print(f"  were the strongest factors!")


# ==============================================================================
# PREDICTION ON NEW PATIENTS
# ==============================================================================

def predict_new_patients(model: nn.Module) -> None:
    """Demonstrate predictions on example patients."""

    print("\n" + "=" * 55)
    print("PREDICTIONS ON NEW PATIENTS")
    print("=" * 55)

    # Example patients: [fever, cough, throat, ache, fatigue, age, contact]
    new_patients = [
        [0.9, 1, 1, 1, 0.8, 0.35, 1],  # Very sick + contact
        [0.1, 0, 0, 0, 0.2, 0.25, 0],  # Healthy
        [0.5, 1, 0, 0, 0.4, 0.65, 1],  # Moderate, elderly, contact
        [0.3, 1, 1, 0, 0.3, 0.20, 0],  # Mild symptoms, no contact
    ]

    descriptions = [
        "High fever, all symptoms, flu contact",
        "No symptoms, no contact (healthy)",
        "Moderate symptoms, elderly, flu contact",
        "Low fever, cough+throat, no contact"
    ]

    patients_tensor = torch.tensor(new_patients, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(patients_tensor)

    print("\n" + "-" * 70)
    for i, (desc, prob) in enumerate(zip(descriptions, predictions)):
        diagnosis = "FLU LIKELY" if prob > 0.5 else "NO FLU"
        print(f"Patient {i+1}: {desc}")
        print(f"           Probability: {prob.item():.1%} -> {diagnosis}")
        print()


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    # =========================================================================
    # PARSE ARGUMENTS
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='Train a neural network to predict flu from patient data'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input CSV file path (default: data/patients.csv)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # =========================================================================
    # DETERMINE INPUT PATH
    # =========================================================================
    if args.input:
        csv_path = Path(args.input)
    else:
        project_root = Path(__file__).parent.parent.parent
        csv_path = project_root / "data" / "patients.csv"

    if not csv_path.exists():
        print(f"\nERROR: Data file not found: {csv_path}")
        print(f"\nPlease generate data first:")
        print(f"  poetry run python -m flu_neural.generate_data")
        return

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n" + "=" * 55)
    print("FLU PREDICTOR - NEURAL NETWORK TRAINING")
    print("=" * 55)

    X, y = load_data(csv_path)

    # =========================================================================
    # SPLIT INTO TRAIN/TEST
    # =========================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} patients")
    print(f"  Testing:  {len(X_test)} patients")

    # =========================================================================
    # CONVERT TO TENSORS
    # =========================================================================
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    print(f"\nTensor shapes:")
    print(f"  X_train: {X_train_t.shape} (patients x features)")
    print(f"  y_train: {y_train_t.shape} (patients x 1)")

    # =========================================================================
    # BUILD MODEL
    # =========================================================================
    print("\n" + "=" * 55)
    print("NEURAL NETWORK ARCHITECTURE")
    print("=" * 55)

    model = FluPredictor(input_size=7, hidden1=16, hidden2=8)
    print(f"\n{model}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params}")

    # =========================================================================
    # TRAIN
    # =========================================================================
    print("\n" + "=" * 55)
    print("TRAINING")
    print("=" * 55)

    loss_history = train_model(
        model, X_train_t, y_train_t,
        epochs=args.epochs,
        learning_rate=args.lr
    )

    # =========================================================================
    # EVALUATE
    # =========================================================================
    evaluate_model(model, X_test_t, y_test_t)

    # =========================================================================
    # ANALYZE
    # =========================================================================
    analyze_feature_importance(model)

    # =========================================================================
    # DEMO PREDICTIONS
    # =========================================================================
    predict_new_patients(model)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print("=" * 55)
    print(f"""
What you can try next:
  1. Edit data/patients.csv and re-run training
  2. Try different hyperparameters:
     --epochs 200 --lr 0.001
  3. Generate more/less data:
     python -m flu_neural.generate_data --samples 5000
""")


if __name__ == "__main__":
    main()
