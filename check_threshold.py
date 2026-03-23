"""
Assignment 5 - check_threshold.py
Reads the Run ID from model_info.txt, fetches accuracy from MLflow,
and exits with code 1 (fails the pipeline) if accuracy is below 0.85.
"""

import sys
import mlflow

THRESHOLD = 0.85

# ── Read Run ID ───────────────────────────────────────────────────────────────
try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Checking Run ID: {run_id}")
except FileNotFoundError:
    print("ERROR: model_info.txt not found.")
    sys.exit(1)

# ── Fetch accuracy from MLflow ────────────────────────────────────────────────
try:
    run = mlflow.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        print("ERROR: 'accuracy' metric not found in MLflow run.")
        sys.exit(1)

    print(f"Accuracy from MLflow: {accuracy:.4f}")
    print(f"Threshold           : {THRESHOLD}")

except Exception as e:
    print(f"ERROR fetching MLflow run: {e}")
    sys.exit(1)

# ── Decision ──────────────────────────────────────────────────────────────────
if accuracy < THRESHOLD:
    print(f"FAILED: accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
    print("Deployment blocked.")
    sys.exit(1)
else:
    print(f"PASSED: accuracy {accuracy:.4f} meets threshold {THRESHOLD}")
    print("Proceeding to deployment.")
    sys.exit(0)
