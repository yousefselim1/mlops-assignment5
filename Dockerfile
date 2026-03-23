# Assignment 5 - Dockerfile
FROM python:3.10-slim

# Accept the MLflow Run ID as a build argument
ARG RUN_ID

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train.py .
COPY check_threshold.py .

# Simulate downloading the model using the Run ID
# In production this would be: mlflow artifacts download --run-id $RUN_ID
RUN echo "Downloading model for Run ID: ${RUN_ID}"

# Set environment variable so the app knows which run to load
ENV MLFLOW_RUN_ID=${RUN_ID}

# Default command
CMD ["python", "-c", "import os; print(f'Model container ready. Run ID: {os.environ.get(\"MLFLOW_RUN_ID\", \"not set\")}')"]
