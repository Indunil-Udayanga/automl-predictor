# 🤖 Smart AutoML System

An intelligent AutoML web application built with Flask and Scikit-learn that automatically detects the target column, identifies whether the dataset is a **Classification** or **Regression** problem, preprocesses the data, trains multiple machine learning models, compares performance, and downloads the best trained model.

## 🚀 Features

* Automatic target column detection
* Automatic problem type detection
* Advanced data preprocessing
* Multiple ML model training & evaluation
* Cross-validation support
* Interactive performance visualizations
* Best model selection & download
* Classification & Regression support

## 🛠️ Tech Stack

* Python
* Flask
* Scikit-learn
* Pandas & NumPy
* Matplotlib & Seaborn

## 📊 Supported Algorithms

### Classification

* Logistic Regression
* Random Forest
* Gradient Boosting
* SVM
* KNN
* Naive Bayes
* MLP Neural Network
* And more...

### Regression

* Linear Regression
* Ridge / Lasso / ElasticNet
* Random Forest Regressor
* Gradient Boosting Regressor
* SVR
* MLP Regressor
* And more...

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## 🌐 Application Workflow

1. Upload dataset (.csv)
2. Auto-detect target & problem type
3. Train multiple ML models
4. Compare performance metrics
5. Download the best trained model

## 📌 Future Improvements

* Hyperparameter tuning
* Deep Learning integration
* Feature importance analysis
* Deployment support
* Explainable AI (XAI)

---
# AutoML Studio 

Automatic machine learning pipeline — upload a CSV, get the best model.

## Folder Structure

```
automl_studio/
├── app/
│   └── main.py          # FastAPI app, ML logic, routes
├── static/
│   ├── css/
│   │   └── style.css    # UI styles
│   └── js/
│       └── app.js       # Frontend JavaScript
├── templates/
│   └── index.html       # Jinja2 HTML template
├── models/              # Saved .pkl model files (auto-created)
├── uploads/             # Temp uploads (auto-created)
├── run.py               # Entry point
└── requirements.txt
```

## Setup & Run

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python run.py
# → http://localhost:8000
```

## Features

| Feature | Detail |
|---|---|
| Auto problem detection | Classifies as regression or classification based on target column |
| Auto target detection  | Finds target column by keyword matching |
| Smart preprocessing    | Missing value imputation, cardinality capping, one-hot encoding |
| 12 algorithms          | Full sklearn model suite for both problem types |
| 5-fold cross-validation| With mean ± std reporting |
| Rich charts            | Bar, CV, Confusion Matrix, Scatter, Radar, Feature Importance, Distribution |
| Model export           | Download best model as `.pkl` with scaler + metadata |
| Analysis tab           | Overfit detection, recommendations, top-5 comparison |

## Using the Downloaded Model

```python
import pickle, pandas as pd

with open("automl_best_model_XXXX.pkl", "rb") as f:
    bundle = pickle.load(f)

model        = bundle["model"]
scaler       = bundle["scaler"]
feature_names = bundle["feature_names"]
problem_type  = bundle["problem_type"]
target_col    = bundle["target_col"]

# Predict on new data
X_new = pd.read_csv("new_data.csv").drop(columns=[target_col])
X_new = pd.get_dummies(X_new).reindex(columns=feature_names, fill_value=0)
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
print(predictions)
```


⭐ If you like this project, give it a star on GitHub!
