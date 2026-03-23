"""
Assignment 5 - train.py
Trains a simple sklearn classifier on the Iris dataset.
Fast (runs in seconds), logs to MLflow, exports Run ID to model_info.txt.

Usage:
    python train.py              # good run  (~0.97 accuracy) -> pipeline PASSES
    python train.py --fail       # bad run   (~0.60 accuracy) -> pipeline FAILS
"""

import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Argument ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--fail", action="store_true",
                    help="Force a low-accuracy run to demo pipeline failure")
args = parser.parse_args()

# ── Data ──────────────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow.set_experiment("Assignment5_YosefSelim")

with mlflow.start_run() as run:
    run_id = run.info.run_id

    if args.fail:
        # Deliberately bad model for failure demo
        model = LogisticRegression(max_iter=1, C=0.00001)
        mlflow.log_param("model_type", "LogisticRegression_intentionally_bad")
        mlflow.log_param("max_iter", 1)
    else:
        # Good model for success demo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.set_tag("student_id", "YosefSelim")
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Run ID  : {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Status  : {'PASS' if accuracy >= 0.85 else 'FAIL'} (threshold=0.85)")

# ── Export Run ID to file ─────────────────────────────────────────────────────
with open("model_info.txt", "w") as f:
    f.write(run_id)

print(f"Run ID saved to model_info.txt")
