from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pickle
import numpy as np
import os
import mlflow

app = FastAPI(
    title="Iris Classification API",
    description="MLOps Pipeline - Iris Flower Classification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODELS_PATH = "/app/models"
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')

# Global model variable
model = None
model_info = None
feature_names = None

# Pydantic models
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: str
    prediction_id: int
    confidence: float
    model_name: str
    model_accuracy: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str]

class ModelInfo(BaseModel):
    name: str
    accuracy: float
    f1_score: float

# Target mapping
TARGET_NAMES = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

def load_model():
    """Load the trained model"""
    global model, model_info, feature_names
    
    model_path = f"{MODELS_PATH}/best_model.pkl"
    info_path = f"{MODELS_PATH}/model_info.pkl"
    features_path = f"{MODELS_PATH}/feature_names.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    
    if os.path.exists(info_path):
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
        print(f"Model info: {model_info}")
    
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"Feature names: {feature_names}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Iris Classification API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_name=model_info.get('name') if model_info else None
    )

@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    if model_info is None:
        raise HTTPException(status_code=404, detail="Model info not available")
    
    return ModelInfo(
        name=model_info.get('name', 'unknown'),
        accuracy=model_info.get('accuracy', 0),
        f1_score=model_info.get('f1_score', 0)
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: IrisInput):
    """Make prediction for iris flower classification"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please run the training pipeline first."
        )
    
    # Prepare input
    features = np.array([[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]])
    
    # Make prediction
    prediction_id = int(model.predict(features)[0])
    prediction_name = TARGET_NAMES.get(prediction_id, 'unknown')
    
    # Get confidence (probability)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
    else:
        confidence = 1.0
    
    return PredictionResponse(
        prediction=prediction_name,
        prediction_id=prediction_id,
        confidence=round(confidence, 4),
        model_name=model_info.get('name', 'unknown') if model_info else 'unknown',
        model_accuracy=round(model_info.get('accuracy', 0), 4) if model_info else 0
    )

@app.post("/reload-model", tags=["Model"])
async def reload_model():
    """Reload the model from disk"""
    load_model()
    
    if model is None:
        raise HTTPException(status_code=404, detail="No model found to load")
    
    return {
        "message": "Model reloaded successfully",
        "model_name": model_info.get('name') if model_info else None
    }

@app.get("/feature-names", tags=["Model"])
async def get_feature_names():
    """Get the feature names used by the model"""
    if feature_names is None:
        return {
            "features": [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)"
            ]
        }
    return {"features": feature_names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
