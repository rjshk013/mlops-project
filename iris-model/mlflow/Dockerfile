FROM python:3.10-slim

WORKDIR /mlflow

RUN pip install mlflow==2.9.2 scikit-learn

RUN mkdir -p /mlflow/mlruns /mlflow/mlartifacts

EXPOSE 5000
