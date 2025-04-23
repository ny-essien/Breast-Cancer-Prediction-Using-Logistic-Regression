import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config):
        """Initialize the DataPreprocessor with configuration."""
        self.config = config
        self.target_column = config['preprocessing']['target_column']
        self.test_size = config['preprocessing']['test_size']
        self.random_state = config['preprocessing']['random_state']
        self.scaler = StandardScaler()
        self.metrics = {
            "data_quality": {
                "missing_values": 0,
                "total_samples": 0,
                "feature_count": 0
            },
            "preprocessing": {
                "train_samples": 0,
                "test_samples": 0,
                "scaling_applied": True
            }
        }

    def preprocess_data(self, df):
        """Preprocess the data and split into train and test sets."""
        try:
            logger.info("Starting data preprocessing...")
            
            # Drop unnecessary columns if they exist
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
            
            # Convert diagnosis to binary (M=1, B=0) if needed
            if df[self.target_column].dtype == 'object':
                df[self.target_column] = df[self.target_column].map({'M': 1, 'B': 0})
            
            # Separate features and target
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Data preprocessing completed. Training set shape: {X_train_scaled.shape}")
            return X_train_scaled, X_test_scaled, y_train, y_test
        
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            raise

    def save_preprocessor(self, path: str = "models/scaler.pkl"):
        """Save the fitted scaler."""
        try:
            import joblib
            import os
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            joblib.dump(self.scaler, path)
            logger.info(f"Preprocessor saved to {path}")
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}")
            raise

    def save_metrics(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save metrics to JSON file."""
        try:
            self.metrics["preprocessing"]["train_samples"] = int(len(train_df))
            self.metrics["preprocessing"]["test_samples"] = int(len(test_df))
            self.metrics["data_quality"]["total_samples"] = int(self.metrics["data_quality"]["total_samples"])
            self.metrics["data_quality"]["missing_values"] = int(self.metrics["data_quality"]["missing_values"])
            self.metrics["data_quality"]["feature_count"] = int(self.metrics["data_quality"]["feature_count"])
            
            with open("metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=4)
            logger.info("Metrics saved to metrics.json")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            raise

def main():
    """Main function to run the data preprocessing pipeline."""
    try:
        # Initialize data loader
        data_loader = DataLoader()
        
        # Load raw data
        raw_data = data_loader.load_raw_data()
        
        # Validate data
        if not data_loader.validate_data(raw_data):
            raise ValueError("Data validation failed")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(data_loader.config)
        
        # Preprocess data
        X_train_scaled, X_test_scaled, y_train, y_test = preprocessor.preprocess_data(raw_data)
        
        # Split data
        train_data, test_data = data_loader.split_data(pd.concat([X_train_scaled, y_train], axis=1))
        
        # Save processed data
        data_loader.save_processed_data(
            train_data,
            data_loader.config['data']['train_path']
        )
        data_loader.save_processed_data(
            test_data,
            data_loader.config['data']['test_path']
        )
        
        # Save preprocessor
        preprocessor.save_preprocessor()
        
        # Save metrics
        preprocessor.save_metrics(train_data, test_data)
        
        logger.info("Data preprocessing pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in data preprocessing pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 