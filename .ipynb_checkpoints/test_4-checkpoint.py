import pytest
import pandas as pd
import numpy as np
import os
import joblib

DATA_PATH = 'data/iris.csv'
ARTIFACTS_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')
EXPECTED_CLASSES = ['setosa', 'versicolor', 'virginica']

#Data Validation
def test_data_validation():
    """
    Tests if the source data file exists and has the expected structure.
    """
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"
    
    df = pd.read_csv(DATA_PATH)
    
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    assert all(col in df.columns for col in expected_columns), "Data is missing expected columns"
    
    # Check if the class names are as expected
    assert set(df['species'].unique()) == set(EXPECTED_CLASSES), "Dataset species names are incorrect"

#Model Sanity Check
def test_model_sanity_check():
    """
    Tests if the model artifact can be loaded and can make a valid prediction.
    """
    assert os.path.exists(MODEL_PATH), f"Model file '{MODEL_PATH}' not found. Please run train.py first."
    
    # Load model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        pytest.fail(f"Failed to load the model from {MODEL_PATH}: {e}")

    # classic setosa
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    
    # Make prediction
    try:
        prediction = model.predict(sample_input)
    except Exception as e:
        pytest.fail(f"Model prediction failed for input {sample_input}: {e}")
    
    
    assert isinstance(prediction, np.ndarray), "Prediction should be a numpy array"
    assert prediction.shape == (1,), "Prediction should have shape (1,)"
    assert prediction[0] in EXPECTED_CLASSES, f"Prediction '{prediction[0]}' is not a valid class"
    print(f"\nModel sanity check passed. Predicted class: {prediction[0]}")