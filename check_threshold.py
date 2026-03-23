"""
Assignment 5 - check_threshold.py
Reads Run ID AND accuracy directly from model_info.txt.
No MLflow server needed across jobs.
"""

import sys

THRESHOLD = 0.85

# ── Read model_info.txt ───────────────────────────────────────────────────────
try:
    with open("model_info.txt", "r") as f:
        lines = f.read().strip().split("\n")
    run_id   = lines[0].strip()
    accuracy = float(lines[1].strip())
    print(f"Run ID  : {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {THRESHOLD}")
except FileNotFoundError:
    print("ERROR: model_info.txt not found.")
    sys.exit(1)
except (IndexError, ValueError) as e:
    print(f"ERROR reading model_info.txt: {e}")
    sys.exit(1)

# ── Decision ──────────────────────────────────────────────────────────────────
if accuracy < THRESHOLD:
    print(f"FAILED: {accuracy:.4f} is below threshold {THRESHOLD}")
    print("Deployment blocked.")
    sys.exit(1)
else:
    print(f"PASSED: {accuracy:.4f} meets threshold {THRESHOLD}")
    print("Proceeding to deployment.")
    sys.exit(0)
