import logging
import joblib
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
import mlflow
import mlflow.sklearn
from typing import Dict, Any
from src.data.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config):
        """Initialize the ModelTrainer with configuration."""
        self.config = config
        self.model = None
        self.metrics = {}
        self.feature_selector = None
        self.best_params = None

    def select_features(self, X_train, y_train):
        """Select the most important features using ANOVA F-value."""
        try:
            logger.info("Selecting most important features...")
            self.feature_selector = SelectKBest(score_func=f_classif, k='all')
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            
            # Get feature scores
            feature_scores = self.feature_selector.scores_
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
            feature_importance = dict(zip(feature_names, feature_scores))
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("Top 5 most important features:")
            for feature, score in sorted_features[:5]:
                logger.info(f"{feature}: {score:.4f}")
            
            return X_train_selected, feature_importance
        except Exception as e:
            logger.error(f"Error during feature selection: {str(e)}")
            raise

    def tune_hyperparameters(self, X_train, y_train):
        """Perform hyperparameter tuning using GridSearchCV."""
        try:
            logger.info("Starting hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            # Initialize GridSearchCV
            grid_search = GridSearchCV(
                LogisticRegression(random_state=self.config['model']['random_state']),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Perform grid search
            grid_search.fit(X_train, y_train)
            
            # Store best parameters
            self.best_params = grid_search.best_params_
            logger.info(f"Best hyperparameters: {self.best_params}")
            
            return grid_search.best_estimator_
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise

    def cross_validate(self, model, X_train, y_train):
        """Perform cross-validation to get robust performance estimates."""
        try:
            logger.info("Performing cross-validation...")
            
            # Perform 5-fold cross-validation
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring='accuracy'
            )
            
            cv_metrics = {
                'mean_accuracy': np.mean(cv_scores),
                'std_accuracy': np.std(cv_scores),
                'cv_scores': cv_scores.tolist()
            }
            
            logger.info(f"Cross-validation results - Mean accuracy: {cv_metrics['mean_accuracy']:.4f} ± {cv_metrics['std_accuracy']:.4f}")
            return cv_metrics
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            raise

    def train_model(self, X_train, y_train):
        """Train the logistic regression model with feature selection and hyperparameter tuning."""
        try:
            logger.info("Starting model training pipeline...")
            
            # Select features
            X_train_selected, feature_importance = self.select_features(X_train, y_train)
            
            # Tune hyperparameters
            best_model = self.tune_hyperparameters(X_train_selected, y_train)
            
            # Perform cross-validation
            cv_metrics = self.cross_validate(best_model, X_train_selected, y_train)
            
            # Train final model
            self.model = best_model
            self.model.fit(X_train_selected, y_train)
            
            # Store feature importance and CV metrics
            self.metrics['feature_importance'] = feature_importance
            self.metrics['cross_validation'] = cv_metrics
            
            logger.info("Model training completed successfully")
            return self.model
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model and calculate metrics."""
        try:
            logger.info("Evaluating model...")
            
            # Transform test data using feature selector
            X_test_selected = self.feature_selector.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_selected)
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            # Add cross-validation metrics if available
            if hasattr(self, 'metrics') and 'cross_validation' in self.metrics:
                metrics['cv_mean_accuracy'] = self.metrics['cross_validation']['mean_accuracy']
                metrics['cv_std_accuracy'] = self.metrics['cross_validation']['std_accuracy']
            
            logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
            return metrics
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise

    def save_model(self, model):
        """Save the trained model and feature selector to disk."""
        try:
            model_path = self.config['model']['model_path']
            selector_path = os.path.join(os.path.dirname(model_path), 'feature_selector.pkl')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model and feature selector
            joblib.dump(model, model_path)
            joblib.dump(self.feature_selector, selector_path)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Feature selector saved to {selector_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def save_metrics(self, metrics):
        """Save model metrics to a JSON file."""
        try:
            metrics_path = os.path.join(
                os.path.dirname(self.config['model']['model_path']),
                'metrics.json'
            )
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise

def main(config):
    """Main function to run the model training pipeline."""
    try:
        # Initialize MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("breast_cancer_prediction")
        
        # Start MLflow run
        with mlflow.start_run():
            # Initialize data loader
            data_loader = DataLoader(config)
            
            # Initialize model trainer
            trainer = ModelTrainer(config)
            
            # Load and preprocess data
            X_train, X_test, y_train, y_test = data_loader.load_data()
            
            # Train model
            model = trainer.train_model(X_train, y_train)
            
            # Evaluate model
            metrics = trainer.evaluate_model(model, X_test, y_test)
            
            # Log parameters and metrics to MLflow
            mlflow.log_params({
                'random_state': config['model']['random_state']
            })
            
            # Flatten and log metrics
            flat_metrics = {}
            for category, category_metrics in metrics.items():
                if isinstance(category_metrics, dict):
                    for metric_name, value in category_metrics.items():
                        if isinstance(value, (int, float)):
                            flat_metrics[f"{category}_{metric_name}"] = value
                        elif isinstance(value, list):
                            # Log each element of the list as a separate metric
                            for i, v in enumerate(value):
                                if isinstance(v, (int, float)):
                                    flat_metrics[f"{category}_{metric_name}_{i}"] = v
                elif isinstance(category_metrics, (int, float)):
                    flat_metrics[category] = category_metrics
            
            mlflow.log_metrics(flat_metrics)
            
            # Save model and metrics
            trainer.save_model(model)
            trainer.save_metrics(metrics)
            
            logger.info("Model training pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise

if __name__ == "__main__":
    # This block is for direct script execution
    from src.config.config_loader import load_config
    config = load_config()
    main(config) 