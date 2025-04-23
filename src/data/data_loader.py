import pandas as pd
import yaml
import os
from typing import Tuple
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config):
        """Initialize the DataLoader with configuration."""
        self.config = config
        self.raw_data_path = config['data']['raw_path']
        self.target_column = config['preprocessing']['target_column']
        self.test_size = config['preprocessing']['test_size']
        self.random_state = config['preprocessing']['random_state']
        self.scaler = StandardScaler()
        self._validate_config()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise

    def _validate_config(self):
        """Validate the configuration."""
        required_paths = [
            self.config['data']['raw_path'],
            self.config['data']['processed_path'],
            self.config['data']['train_path'],
            self.config['data']['test_path']
        ]
        
        for path in required_paths:
            dir_path = os.path.dirname(path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"Created directory: {dir_path}")

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess the data."""
        try:
            logger.info(f"Loading data from {self.raw_data_path}")
            if not os.path.exists(self.raw_data_path):
                raise FileNotFoundError(f"Data file not found at {self.raw_data_path}")
            
            # Load raw data
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Drop ID column if it exists
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
            logger.error(f"Error loading data: {str(e)}")
            raise

    def save_processed_data(self, df: pd.DataFrame, path: str):
        """Save processed data to specified path."""
        try:
            logger.info(f"Saving processed data to {path}")
            df.to_csv(path, index=False)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the dataset for required columns and data quality."""
        try:
            # Check for missing values
            if df.isnull().sum().sum() > 0:
                logger.warning("Dataset contains missing values")
                return False

            # Check for required columns
            required_columns = ['id', 'diagnosis']  # Add other required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check for data types
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_columns) < 30:  # Assuming we expect at least 30 numeric features
                logger.warning("Unexpected number of numeric columns")
                return False

            logger.info("Data validation passed")
            return True
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            return False

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets."""
        try:
            from sklearn.model_selection import train_test_split
            
            X = df.drop(columns=[self.config['preprocessing']['target_column']])
            y = df[self.config['preprocessing']['target_column']]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['preprocessing']['test_size'],
                random_state=self.config['preprocessing']['random_state']
            )
            
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
            
            logger.info(f"Data split into train: {train_df.shape} and test: {test_df.shape}")
            return train_df, test_df
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise 