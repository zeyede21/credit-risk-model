import sys
import os
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import preprocess_features

def test_preprocess_features_columns():
    df = pd.DataFrame({
        "numeric_col": [1, 2, 3],
        "category_col": ["a", "b", "a"]
    })
    X, pipeline = preprocess_features(df)
    assert X.shape[0] == 3

def test_preprocess_features_shape():
    df = pd.DataFrame({
        "numeric_col": [1, 2, 3],
        "category_col": ["a", "b", "a"]
    })
    X, pipeline = preprocess_features(df)
    assert X.shape[0] == 3
