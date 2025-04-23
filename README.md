# Breast Cancer Prediction System

This project implements a machine learning system for predicting breast cancer diagnosis using logistic regression. It includes a FastAPI web interface for making predictions.

## Features

- Logistic Regression model for breast cancer prediction
- Feature selection using ANOVA F-value
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust performance estimation
- FastAPI web interface for predictions
- MLflow integration for experiment tracking

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd breast-cancer-prediction
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python src/models/train.py
```

2. Start the FastAPI server:
```bash
uvicorn src.api.app:app --reload
```

3. Access the web interface:
Open your browser and navigate to `http://localhost:8000`

## Project Structure

```
breast-cancer-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── model
│   └── feature_selector.pkl
├── src/
│   ├── api/
│   │   ├── app.py
│   │   └── templates/
│   │       └── index.html
│   ├── data/
│   │   └── data_loader.py
│   └── models/
│       └── train.py
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## API Endpoints

- `GET /`: Main prediction interface
- `POST /predict`: Make predictions
- `GET /model_info`: Get model information

## License

This project is licensed under the MIT License - see the LICENSE file for details. 