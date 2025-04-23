import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.train import main
from src.config.config_loader import load_config

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Run the training pipeline
    main(config) 