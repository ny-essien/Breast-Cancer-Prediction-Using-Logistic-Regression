import yaml
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        # Get the absolute path to the config file
        abs_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), config_path)
        
        with open(abs_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded successfully from {abs_config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise 