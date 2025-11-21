# MLOps Iris Classification Pipeline 🌸

A complete end-to-end MLOps pipeline with Airflow, MLflow, FastAPI, and a web frontend.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Airflow   │ ──▶ │   MLflow    │ ──▶ │  FastAPI    │ ──▶ │  Frontend   │
│   :8080     │     │   :5000     │     │   :8000     │     │   :3000     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
   Schedule           Track              Serve               Display
   Training          Experiments        Predictions          Results
```

## Services

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Frontend** | 3000 | http://localhost:3000 | Web UI for predictions |
| **FastAPI** | 8000 | http://localhost:8000/docs | REST API |
| **MLflow** | 5000 | http://localhost:5000 | Experiment tracking |
| **Airflow** | 8080 | http://localhost:8080 | Pipeline orchestration |

## Quick Start

### Step 1: Clone/Create Project

```bash
cd mlops-iris-pipeline
```

### Step 2: Create Required Directories

```bash
mkdir -p models data mlflow/mlruns mlflow/mlartifacts
```

### Step 3: Start All Services

```bash
docker-compose up -d --build
```

Wait 2-3 minutes for all services to start.

### Step 4: Access Services

- **Frontend**: http://localhost:3000
- **Airflow**: http://localhost:8080 (admin/admin)
- **MLflow**: http://localhost:5000
- **API Docs**: http://localhost:8000/docs

### Step 5: Run Training Pipeline

1. Open Airflow: http://localhost:8080
2. Login: `admin` / `admin`
3. Find DAG: `iris_ml_pipeline`
4. Toggle ON (enable the DAG)
5. Click ▶ (Trigger DAG)
6. Wait for pipeline to complete (green)

### Step 6: Make Predictions

1. Open Frontend: http://localhost:3000
2. Enter flower measurements
3. Click "Predict Species"
4. See the result!

## Pipeline Workflow

The Airflow DAG runs 4 tasks:

```
[load_data] → [preprocess_data] → [train_models] → [register_model]
```

1. **load_data**: Downloads Iris dataset
2. **preprocess_data**: Splits into train/test
3. **train_models**: Trains 3 models, logs to MLflow
4. **register_model**: Saves best model to registry

## Models Trained

- Random Forest (100 trees)
- Logistic Regression
- Random Forest Tuned (200 trees, max_depth=10)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check API status |
| GET | `/model-info` | Get loaded model info |
| POST | `/predict` | Make prediction |
| POST | `/reload-model` | Reload model from disk |

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Response

```json
{
  "prediction": "setosa",
  "prediction_id": 0,
  "confidence": 1.0,
  "model_name": "random_forest",
  "model_accuracy": 1.0
}
```

## Project Structure

```
mlops-iris-pipeline/
├── docker-compose.yaml      # All services
├── airflow/
│   ├── Dockerfile
│   └── dags/
│       └── iris_training_dag.py
├── mlflow/
│   ├── Dockerfile
│   ├── mlruns/              # Experiment data
│   └── mlartifacts/         # Model artifacts
├── backend/
│   ├── Dockerfile
│   └── main.py              # FastAPI app
├── frontend/
│   ├── Dockerfile
│   └── index.html           # Web UI
├── models/                   # Saved models
├── data/                     # Datasets
└── README.md
```

## Useful Commands

```bash
# Start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f airflow
docker-compose logs -f backend

# Stop all services
docker-compose down

# Restart a service
docker-compose restart backend

# Rebuild a specific service
docker-compose up -d --build backend

# Check running containers
docker ps

# Enter a container
docker exec -it backend bash
```

## Troubleshooting

### Model not loaded error

Run the Airflow pipeline first to train and save a model.

### Airflow not starting

Wait 2-3 minutes for database initialization.

### Cannot connect to API

Check if backend container is running:
```bash
docker-compose logs backend
```

### MLflow not showing experiments

1. Check MLflow container is running
2. Verify tracking URI in Airflow DAG

## Customization

### Change Training Schedule

Edit `airflow/dags/iris_training_dag.py`:

```python
# Daily at 6 AM
schedule_interval='0 6 * * *'

# Every hour
schedule_interval='@hourly'

# Manual only
schedule_interval=None
```

### Add New Model

In `iris_training_dag.py`, add to models dict:

```python
models = {
    ...
    'xgboost': XGBClassifier(n_estimators=100),
}
```

### Change Dataset

Modify `load_data()` function to load your dataset.

## What You'll Learn

- ✅ Airflow DAG creation and scheduling
- ✅ MLflow experiment tracking
- ✅ MLflow model registry
- ✅ FastAPI REST endpoints
- ✅ Docker multi-container orchestration
- ✅ Frontend-backend integration
- ✅ End-to-end ML pipeline

## Next Steps

- [ ] Add data drift monitoring (Evidently AI)
- [ ] Add CI/CD with GitHub Actions
- [ ] Deploy to Kubernetes
- [ ] Add authentication
- [ ] Add more models (XGBoost, LightGBM)

## License

MIT License
