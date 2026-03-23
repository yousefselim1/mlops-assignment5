"""
Assignment 5 - train.py
Trains a simple sklearn classifier on the Iris dataset.
Logs to MLflow, exports Run ID AND accuracy to model_info.txt.

Usage:
    python train.py              # good run (~0.97 accuracy) -> pipeline PASSES
    python train.py --fail       # bad run  (~0.60 accuracy) -> pipeline FAILS
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
        model = LogisticRegression(max_iter=1, C=0.00001)
        mlflow.log_param("model_type", "LogisticRegression_intentionally_bad")
        mlflow.log_param("max_iter", 1)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.set_tag("student_id", "YosefSelim")
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Run ID  : {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Status  : {'PASS' if accuracy >= 0.85 else 'FAIL'} (threshold=0.85)")

# ── Export Run ID AND accuracy to file ────────────────────────────────────────
# Both values saved so deploy job can check accuracy without MLflow server
with open("model_info.txt", "w") as f:
    f.write(f"{run_id}\n{accuracy}")

print(f"Saved to model_info.txt: run_id={run_id}, accuracy={accuracy:.4f}")
