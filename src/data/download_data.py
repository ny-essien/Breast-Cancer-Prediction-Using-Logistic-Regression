import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def download_breast_cancer_dataset():
    """Download the breast cancer dataset from scikit-learn and save it as CSV."""
    # Load the dataset
    breast_cancer = load_breast_cancer()
    
    # Create a DataFrame
    data = pd.DataFrame(
        breast_cancer.data,
        columns=breast_cancer.feature_names
    )
    
    # Add the target column
    data['diagnosis'] = breast_cancer.target
    
    # Add an ID column
    data.insert(0, 'id', range(len(data)))
    
    # Create the output directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Save the dataset
    output_path = 'data/raw/breast_cancer.csv'
    data.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    download_breast_cancer_dataset() 