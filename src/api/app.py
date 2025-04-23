from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import joblib
import os
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for predicting breast cancer diagnosis using logistic regression",
    version="1.0.0"
)

# Set up templates
templates = Jinja2Templates(directory="src/api/templates")

# Load model and feature selector
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'model')
selector_path = os.path.join(os.path.dirname(model_path), 'feature_selector.pkl')

try:
    model = joblib.load(model_path)
    feature_selector = joblib.load(selector_path)
    logger.info("Model and feature selector loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

class Features(BaseModel):
    """Input features for prediction."""
    features: List[float]

class Prediction(BaseModel):
    """Prediction response."""
    prediction: int
    probability: float
    diagnosis: str

@app.get("/")
async def root(request: Request):
    """Serve the main prediction interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=Prediction)
async def predict(features: Features):
    """Make a prediction using the trained model."""
    try:
        # Convert features to numpy array
        X = np.array(features.features).reshape(1, -1)
        
        # Transform features using feature selector
        X_selected = feature_selector.transform(X)
        
        # Make prediction
        prediction = model.predict(X_selected)[0]
        probability = model.predict_proba(X_selected)[0][1]
        
        # Convert prediction to diagnosis
        diagnosis = "Malignant" if prediction == 1 else "Benign"
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "diagnosis": diagnosis
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def model_info():
    """Get information about the trained model."""
    try:
        return {
            "model_type": type(model).__name__,
            "feature_count": feature_selector.n_features_in_,
            "selected_features": feature_selector.get_support().tolist()
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 