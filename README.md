# Credit Risk Model Using Alternative Data

A machine learning pipeline to predict high-risk customers using e-commerce behavioral data. Built with FastAPI, MLflow, Docker, and CI/CD.

---

## 🚀 Project Overview

This project, developed for Bati Bank, predicts customer credit risk based on e-commerce behavioral data. It includes exploratory data analysis (EDA), RFM-based proxy label creation, ML model training with experiment tracking via MLflow, API deployment with FastAPI, and CI/CD automation with GitHub Actions.

---

## 📂 Project Structure

```
credit-risk-model/
├── data/                  # Raw and processed data
├── src/                   # Source code for data processing, training, and API
│   ├── api/               # FastAPI app
│   │   ├── main.py
│   │   └── pydantic_models.py
│   └── train.py           # Model training script
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container setup for FastAPI
├── docker-compose.yml     # Compose file to run app
├── .github/workflows/ci.yml # GitHub Actions CI/CD workflow
└── README.md
```

---

## 🧪 How to Run

### 1. Setup Environment

```bash
python -m venv myenvv
source myenvv/bin/activate  # or myenvv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run MLflow UI

```bash
mlflow ui --port 5000
```

Visit: [http://localhost:5000](http://localhost:5000)

### 3. Train Models

```bash
python src/train.py
```

Models and metrics will be logged to MLflow.

### 4. Start API

```bash
uvicorn src.api.main:app --reload
```

Visit Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Run Tests & Lint

```bash
pytest tests/
flake8 .
```

---

## 🐳 Docker Setup

### Build and Run

```bash
docker-compose up --build
```

---

## 🧠 Features

- RFM clustering for proxy label creation
- Logistic Regression and Random Forest models
- MLflow model tracking and registry
- FastAPI with Pydantic validation
- Dockerized API for deployment
- GitHub Actions for CI (lint + test)

---

## 📸 Screenshots

- MLflow UI showing experiments
  ![alt text](<Screenshot 2025-07-01 142559.png>)
- Registered models
  ![alt text](<Screenshot 2025-07-01 142746.png>)
- FastAPI Swagger UI
  ![alt text](<Screenshot 2025-06-28 180345.png>)
- Passing CI pipeline
  ![alt text](<Screenshot 2025-07-01 152109.png>)
- Docker container running
  ![alt text](<Screenshot 2025-06-28 180923.png>)

---

## ✍️ Authors

- Zeyede
