# =============================================================================
# run_pipeline.py
# -----------------------------------------------------------------------------
# This is the ORCHESTRATOR — the script you run manually.
# It decides which CSV parts still need to be processed, then calls train.py
# once for each unprocessed file.
#
# Key idea: it keeps a plain-text log (processed_files.txt) of every CSV it
# has already run. That way, if you add a new CSV part later and run this
# script again, it will ONLY process the new file — it won't re-train on
# files it has already seen.
#
# Usage:
#   python run_pipeline.py
# =============================================================================

import os
import subprocess
import sys


# -----------------------------------------------------------------------------
# CONFIGURATION — edit these two settings to control the pipeline
# -----------------------------------------------------------------------------

# The ordered list of CSV files the pipeline should process.
# Add "part3.csv", "part4.csv", etc. here as you get more data.
CSV_PARTS = [
    "part1.csv",
    "part2.csv",
]

# The training script that will be called for each CSV file.
# Each call is a completely separate process, mirroring a real pipeline step.
PIPELINE_SCRIPT = "train.py"

# A plain-text file that records which CSV parts have already been processed.
# Each line is one filename. New runs skip files already listed here.
PROCESSED_LOG = "processed_files.txt"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_processed_files():
    """
    Read processed_files.txt and return a set of filenames already done.

    Using a set gives O(1) lookup — fast to check "have we seen this file?"
    Returns an empty set if the log file doesn't exist yet (first ever run).
    """
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r") as f:
            return set(line.strip() for line in f.readlines())
    return set()


def mark_as_processed(csv_file):
    """
    Append a filename to processed_files.txt after a successful training run.

    Uses "a" (append) mode so previous entries are never overwritten.
    Only called on SUCCESS — failed runs are NOT marked, so they will be
    retried the next time run_pipeline.py is executed.
    """
    with open(PROCESSED_LOG, "a") as f:
        f.write(csv_file + "\n")


# =============================================================================
# MAIN PIPELINE LOOP
# =============================================================================

def run_pipeline():
    print("=" * 55)
    print("  Starting Automated Titanic MLflow Pipeline")
    print("=" * 55)

    # Find out which files have already been trained on
    processed = load_processed_files()

    # Filter CSV_PARTS down to only files not yet processed
    new_files = [f for f in CSV_PARTS if f not in processed]

    # --- Nothing to do -------------------------------------------------------
    if not new_files:
        print("\n  No new files to process. All parts already ran.")
        print("  Add a new file to CSV_PARTS to run again.")
        print("=" * 55)
        return

    # --- Process each new file -----------------------------------------------
    for i, csv_file in enumerate(new_files, start=1):

        # Safety check: make sure the file actually exists on disk
        if not os.path.exists(csv_file):
            print(f"\n[ERROR] {csv_file} not found, skipping...")
            continue

        print(f"\n[{i}/{len(new_files)}] Running pipeline on {csv_file}...")
        print("-" * 55)

        # Launch train.py as a subprocess, passing the CSV path as an argument.
        # Using subprocess.run (instead of importing train.py directly) keeps
        # each run isolated — MLflow gets its own process, no state leaks
        # between runs.
        # capture_output=False means train.py's print statements appear live
        # in your terminal as the run progresses.
        result = subprocess.run(
            [sys.executable, PIPELINE_SCRIPT, csv_file],
            capture_output=False
        )

        # Only mark a file as done if the training script exited cleanly.
        # returncode 0 = success; anything else = some error occurred.
        if result.returncode == 0:
            print(f"[OK] {csv_file} completed successfully.")
            mark_as_processed(csv_file)   # record it so we skip it next time
        else:
            print(f"[FAILED] {csv_file} failed with return code {result.returncode}.")
            # File is NOT marked processed, so it will be retried next run

    # --- Done ----------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  All runs complete!")
    print("  Open MLflow UI: run 'mlflow ui' then go to")
    print("  http://localhost:5000")
    print("=" * 55)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_pipeline()
