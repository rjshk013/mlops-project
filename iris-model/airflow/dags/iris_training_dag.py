from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import pickle
import os

# Default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MODELS_PATH = '/opt/airflow/models'
DATA_PATH = '/opt/airflow/data'

def load_data(**kwargs):
    """Load Iris dataset and save to disk"""
    print("Loading Iris dataset...")
    
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Save dataset
    os.makedirs(DATA_PATH, exist_ok=True)
    df.to_csv(f'{DATA_PATH}/iris_data.csv', index=False)
    
    print(f"Dataset saved: {df.shape[0]} samples, {df.shape[1]} features")
    return f'{DATA_PATH}/iris_data.csv'

def preprocess_data(**kwargs):
    """Preprocess data and create train/test split"""
    print("Preprocessing data...")
    
    # Load data
    df = pd.read_csv(f'{DATA_PATH}/iris_data.csv')
    
    # Split features and target
    X = df.drop(columns=['target', 'target_name'])
    y = df['target']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save preprocessed data
    os.makedirs(MODELS_PATH, exist_ok=True)
    np.save(f'{MODELS_PATH}/X_train.npy', X_train.values)
    np.save(f'{MODELS_PATH}/X_test.npy', X_test.values)
    np.save(f'{MODELS_PATH}/y_train.npy', y_train.values)
    np.save(f'{MODELS_PATH}/y_test.npy', y_test.values)
    
    # Save feature names
    with open(f'{MODELS_PATH}/feature_names.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

def train_models(**kwargs):
    """Train multiple models and log to MLflow"""
    print("Training models...")
    
    # Load preprocessed data
    X_train = np.load(f'{MODELS_PATH}/X_train.npy')
    X_test = np.load(f'{MODELS_PATH}/X_test.npy')
    y_train = np.load(f'{MODELS_PATH}/y_train.npy')
    y_test = np.load(f'{MODELS_PATH}/y_test.npy')
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("iris-classification")
    
    # Models to train
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=200, random_state=42),
        'random_forest_tuned': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    }
    
    best_accuracy = 0
    best_model_name = None
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Log parameters
            if hasattr(model, 'n_estimators'):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, 'max_depth'):
                mlflow.log_param("max_depth", model.max_depth)
            if hasattr(model, 'max_iter'):
                mlflow.log_param("max_iter", model.max_iter)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                
                # Save best model locally for FastAPI
                with open(f'{MODELS_PATH}/best_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                
                # Save model info
                with open(f'{MODELS_PATH}/model_info.pkl', 'wb') as f:
                    pickle.dump({
                        'name': name,
                        'accuracy': accuracy,
                        'f1_score': f1
                    }, f)
    
    print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

def register_best_model(**kwargs):
    """Register the best model in MLflow Model Registry"""
    print("Registering best model...")
    
    # Load model info
    with open(f'{MODELS_PATH}/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Find the best run
    experiment = mlflow.get_experiment_by_name("iris-classification")
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
        
        # Register model
        model_uri = f"runs:/{best_run['run_id']}/model"
        result = mlflow.register_model(model_uri, "iris-classifier")
        
        print(f"Model registered: iris-classifier version {result.version}")
        print(f"Accuracy: {model_info['accuracy']:.4f}")

# Create DAG
dag = DAG(
    'iris_ml_pipeline',
    default_args=default_args,
    description='Iris ML Training Pipeline',
    schedule_interval=timedelta(days=1),  # Run daily
    catchup=False,
    tags=['mlops', 'training']
)

# Define tasks
task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

task_train = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag
)

task_register = PythonOperator(
    task_id='register_model',
    python_callable=register_best_model,
    dag=dag
)

# Set task dependencies
task_load_data >> task_preprocess >> task_train >> task_register
