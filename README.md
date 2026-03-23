# Assignment 5 — Multi-Job ML Pipeline with GitHub Actions

## Overview
A two-job GitHub Actions pipeline that validates a model and deploys it
only if accuracy meets the required threshold (>= 0.85).

## Files
- `train.py` — trains Iris classifier, logs to MLflow, saves Run ID
- `check_threshold.py` — reads Run ID, checks accuracy, fails if below 0.85
- `requirements.txt` — Python dependencies
- `Dockerfile` — container definition accepting RUN_ID as build arg
- `.github/workflows/pipeline.yml` — the two-job CI/CD pipeline

## Student
Yosef Selim | MLOps Course
