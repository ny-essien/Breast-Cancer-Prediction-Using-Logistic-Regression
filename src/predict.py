import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_path: str):
        """Initialize the ModelPredictor with the path to the trained model."""
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the trained model from disk."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using the loaded model."""
        try:
            predictions = self.model.predict(features)
            logger.info("Predictions made successfully")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities using the loaded model."""
        try:
            probabilities = self.model.predict_proba(features)
            logger.info("Prediction probabilities calculated successfully")
            return probabilities
        except Exception as e:
            logger.error(f"Error calculating prediction probabilities: {str(e)}")
            raise

def main():
    """Main function to demonstrate model prediction."""
    try:
        # Initialize predictor
        model_path = "models/model"  # Update this path as needed
        predictor = ModelPredictor(model_path)

        # Example: Create some sample features for prediction
        # Replace this with your actual feature data
        sample_features = np.array([
            [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871],
            [20.57, 17.77, 132.9, 1326.0, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667]
        ])

        # Make predictions
        predictions = predictor.predict(sample_features)
        probabilities = predictor.predict_proba(sample_features)

        # Print results
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Prediction: {'Malignant' if pred == 1 else 'Benign'}")
            logger.info(f"  Probability of being malignant: {probs[1]:.4f}")
            logger.info(f"  Probability of being benign: {probs[0]:.4f}")

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 