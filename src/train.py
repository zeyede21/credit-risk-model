import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

def preprocess_data(df):
    # Check for high cardinality columns
    high_cardinality_cols = [col for col in df.select_dtypes(include=['object']) if df[col].nunique() > 100]  # Adjust threshold as needed
    
    # Group infrequent categories into 'Other'
    for col in high_cardinality_cols:
        counts = df[col].value_counts()
        df[col] = np.where(df[col].isin(counts[counts > 10].index), df[col], 'Other')  # Adjust frequency threshold as needed
    
    # Convert categorical variables to numerical using one-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    return df

def load_data(path):
    df = pd.read_csv(path)
    df = preprocess_data(df)  # Preprocess the data
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }

def train_and_log(model, params, X_train, y_train, X_test, y_test, model_name="credit_risk_model"):
    with mlflow.start_run() as run:
        grid = GridSearchCV(model, param_grid=params, cv=3, scoring='f1')
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)

        print("\nModel Performance:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # Log the model and register it
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=model_name  # ⬅️ This registers the model
        )

        print(f"🏷️ Registered model: {model_name}")
        print(f"🔗 View run at: http://localhost:5000/#/experiments/0/runs/{run.info.run_id}")

        return best_model


def main():
    X_train, X_test, y_train, y_test = load_data("D:/10Academy1/credit-risk-model/data/processed/preprocess_dataset.csv")
    
    print("\n🔁 Training Logistic Regression...")
    train_and_log(LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}, X_train, y_train, X_test, y_test)

    print("\n🌲 Training Random Forest...")
    train_and_log(RandomForestClassifier(), {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 10]
    }, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()