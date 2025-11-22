FROM python:3.10-slim

RUN pip install mlflow==2.9.2 scikit-learn

# Create working directory with proper permissions
RUN mkdir -p /mlflow/mlartifacts && chmod -R 777 /mlflow

WORKDIR /mlflow

EXPOSE 5000

# Use file-based backend in working directory
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "./mlartifacts"]

