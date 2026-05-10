# =============================================================================
# train.py
# -----------------------------------------------------------------------------
# This is the core ML pipeline script. It is called by run_pipeline.py and
# receives one argument: the path to a CSV data file (e.g. part1.csv).
#
# What it does, end to end:
#   1. Loads and cleans the CSV data
#   2. Trains a Random Forest classifier on 80% of that data
#   3. Evaluates accuracy on the remaining 20%
#   4. Appends predictions from this run to a running cumulative CSV
#   5. Logs everything (params, metrics, model, predictions) to MLflow
# =============================================================================

import os

# Suppress noisy GitPython warnings that MLflow can trigger
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import pandas as pd
import mlflow
import mlflow.sklearn
import tempfile

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

# Every time a new CSV part is trained, its predictions are appended here.
# This file grows with each run so you always have a full history of every
# prediction ever made across all pipeline runs.
CUMULATIVE_PREDICTIONS_PATH = "all_predictions.csv"


# =============================================================================
# STEP 1 — DATA PREPARATION
# =============================================================================

def dataPrep(DATA_PATH):
    """
    Load and prepare a Titanic CSV file for training.

    Steps:
      - Drop columns that are too unique or sparse to be useful features
        (Name, Ticket, Cabin)
      - Drop any rows that still have missing values
      - Separate features (X) from the target label (y = Survived)
      - One-hot encode any remaining categorical columns (e.g. Sex, Embarked)
      - Split into 80% training / 20% test sets (fixed random seed for
        reproducibility)

    Parameters
    ----------
    DATA_PATH : str
        Path to the CSV file to load (e.g. "part1.csv")

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
    """

    df = pd.read_csv(DATA_PATH)

    # These columns are dropped because:
    #   Name   — unique string per passenger, adds no signal
    #   Ticket — alphanumeric code with no consistent pattern
    #   Cabin  — ~77% missing in the original Titanic dataset
    drop_cols = ["Name", "Ticket", "Cabin"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Drop rows with any remaining NaN values (e.g. missing Age or Embarked)
    df = df.dropna()

    # Split into features and target label
    X = df.drop("Survived", axis=1)   # everything except the label
    y = df["Survived"]                 # 0 = did not survive, 1 = survived

    # Convert categorical text columns (Sex, Embarked) into numeric 0/1 columns
    # e.g. "Sex" becomes "Sex_female" and "Sex_male"
    X = pd.get_dummies(X)

    # 80/20 train-test split; random_state=42 ensures the same split every run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 2 — CUMULATIVE PREDICTION STORE (load + save)
# =============================================================================

def load_cumulative_predictions():
    """
    Load the cumulative predictions CSV from previous runs.

    If no previous runs exist yet (first ever run), return an empty DataFrame
    so the rest of the code can treat both cases the same way.
    """
    if os.path.exists(CUMULATIVE_PREDICTIONS_PATH):
        return pd.read_csv(CUMULATIVE_PREDICTIONS_PATH)
    return pd.DataFrame()


def save_cumulative_predictions(df):
    """
    Overwrite the cumulative predictions file with the latest full history.

    This is called after every successful run so the file always reflects
    every prediction made across all parts processed so far.
    """
    df.to_csv(CUMULATIVE_PREDICTIONS_PATH, index=False)


# =============================================================================
# STEP 3 — TRAINING + MLFLOW LOGGING
# =============================================================================

def train(new_data_path):
    """
    Run the full training pipeline for one CSV file and log results to MLflow.

    Parameters
    ----------
    new_data_path : str
        Path to the new CSV file to train on (e.g. "part2.csv")
    """

    # --- Prepare data --------------------------------------------------------
    X_train, X_test, y_train, y_test = dataPrep(new_data_path)

    # --- Load any predictions already made in prior runs ---------------------
    old_predictions = load_cumulative_predictions()

    # --- Start an MLflow run -------------------------------------------------
    # Everything inside this block is tracked as a single MLflow experiment run
    with mlflow.start_run(run_name="Titanic_Run"):

        # Tag the run with which data file was used, so you can filter in the UI
        mlflow.set_tag("data_version", os.path.basename(new_data_path))

        # --- Train model -----------------------------------------------------
        # RandomForestClassifier: an ensemble of decision trees that vote on
        # the final prediction. random_state=42 keeps results deterministic.
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # --- Evaluate --------------------------------------------------------
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)  # fraction of correct predictions

        # --- Build a predictions DataFrame for this run ----------------------
        # We attach metadata columns so you can trace each prediction back to
        # the file it came from and the MLflow run that produced it
        new_predictions = X_test.copy()
        new_predictions["Actual"]      = y_test.values               # ground truth
        new_predictions["Predicted"]   = preds                        # model output
        new_predictions["Source_File"] = os.path.basename(new_data_path)
        new_predictions["Run_ID"]      = mlflow.active_run().info.run_id

        # --- Merge with history ----------------------------------------------
        # If there are predictions from earlier runs, stack them on top so
        # all_predictions.csv always holds the full picture
        if not old_predictions.empty:
            cumulative_predictions = pd.concat(
                [old_predictions, new_predictions], ignore_index=True
            )
        else:
            cumulative_predictions = new_predictions   # first run ever

        # --- Console summary -------------------------------------------------
        print(f"\n{'='*55}")
        print(f"  Total predictions so far: {len(cumulative_predictions)}")
        print(f"  From this run:            {len(new_predictions)}")
        if not old_predictions.empty:
            print(f"  From previous runs:       {len(old_predictions)}")
        print(f"{'='*55}\n")

        print("All Predictions (old + new):")
        print(cumulative_predictions[["Actual", "Predicted", "Source_File"]].to_string())

        # --- Save cumulative predictions locally -----------------------------
        save_cumulative_predictions(cumulative_predictions)

        # --- Log to MLflow ---------------------------------------------------
        # Params: descriptive, non-numeric values about the run setup
        mlflow.log_param("model",                    "RandomForest")
        mlflow.log_param("new_data_file",            os.path.basename(new_data_path))
        mlflow.log_param("total_predictions_so_far", len(cumulative_predictions))

        # Metrics: numeric values MLflow can plot and compare across runs
        mlflow.log_metric("accuracy", acc)

        # Artifact: the full cumulative predictions CSV, stored inside MLflow
        # so every run has a snapshot of predictions up to that point
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = os.path.join(tmp_dir, "all_predictions.csv")
            cumulative_predictions.to_csv(artifact_path, index=False)
            mlflow.log_artifact(artifact_path, artifact_path="predictions")

        # Log the trained model itself so it can be loaded and served later
        mlflow.sklearn.log_model(model, "model")

        print(f"\nAccuracy this run : {acc:.4f}")
        print(f"Run ID            : {mlflow.active_run().info.run_id}")
        print("Cumulative predictions logged to MLflow under predictions/")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    # This script expects exactly one argument: the CSV file path
    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_new_csv>")
        print("Example: python train.py part1.csv")
        sys.exit(1)

    new_csv = sys.argv[1]

    if not os.path.exists(new_csv):
        print(f"Error: File not found -> {new_csv}")
        sys.exit(1)

    train(new_csv)
