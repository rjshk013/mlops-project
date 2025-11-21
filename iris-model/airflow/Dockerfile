FROM apache/airflow:2.7.3-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    scikit-learn \
    pandas \
    numpy \
    xgboost \
    lightgbm

WORKDIR /opt/airflow
