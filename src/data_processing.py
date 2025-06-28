# Placeholder for data_processing.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os

def load_data(path):
    return pd.read_csv(path)

def build_features(df):
    # Convert TransactionStartTime to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Extract date features
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionWeekday'] = df['TransactionStartTime'].dt.weekday

    # Aggregate features per customer
    agg_df = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std', 'count'],
        'Value': ['sum', 'mean', 'std'],
        'FraudResult': 'sum',
        'ProductCategory': pd.Series.nunique,
        'TransactionStartTime': lambda x: (pd.to_datetime(df['TransactionStartTime']).max() - x.max()).days
    }).reset_index()

    agg_df.columns = ['CustomerId',
                      'Amount_sum', 'Amount_mean', 'Amount_std', 'Transaction_count',
                      'Value_sum', 'Value_mean', 'Value_std',
                      'Fraud_count', 'Unique_categories', 'Recency']

    return agg_df

def preprocess_features(df):
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'CustomerId' in numeric_features:numeric_features.remove('CustomerId')  # exclude ID


    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
        ],
        remainder='passthrough'  # keep CustomerId
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    return pipeline.fit_transform(df), pipeline

def save_processed_data(X, customer_ids, path):
    processed_df = pd.DataFrame(X)
    processed_df.insert(0, 'CustomerId', customer_ids.values)
    processed_df.to_csv(path, index=False)

def run_pipeline():
    input_path = r"D:\10Academy1\credit-risk-model\data\processed\preprocess_dataset.csv"
    output_path =  r"D:\10Academy1\credit-risk-model\data\processed\engineered_dataset.csv"

    df = load_data(input_path)
    features_df = build_features(df)
    X, pipeline = preprocess_features(features_df)
    save_processed_data(X, features_df['CustomerId'], output_path)

if __name__ == "__main__":
    run_pipeline()
