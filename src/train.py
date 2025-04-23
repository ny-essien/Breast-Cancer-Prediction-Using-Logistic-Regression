import logging
import yaml
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train import ModelTrainer
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def main():
    """Main function to run the training pipeline."""
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Initialize components
        data_loader = DataLoader(config)
        preprocessor = DataPreprocessor(config)
        trainer = ModelTrainer(config)

        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = data_loader.load_data()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)

        # Train and evaluate model
        logger.info("Training model...")
        model = trainer.train_model(X_train, y_train)
        metrics = trainer.evaluate_model(model, X_test, y_test)

        # Save model and metrics
        logger.info("Saving model and metrics...")
        trainer.save_model(model)
        trainer.save_metrics(metrics)

        logger.info("Training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 