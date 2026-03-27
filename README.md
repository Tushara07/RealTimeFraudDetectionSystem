# Real-Time Fraud Detection System (MLOps Pipeline)

An end-to-end Machine Learning and MLOps project for detecting fraudulent financial transactions using the IEEE-CIS dataset.

This project implements a complete workflow including data preprocessing, model training, experiment tracking, deployment, and logging.

---

## Features

* Data preprocessing and feature engineering
* Model training using XGBoost
* Model comparison and selection (Logistic Regression, Random Forest, XGBoost)
* Model explainability using SHAP
* Interactive dashboard using Streamlit
* CI/CD pipeline using GitHub Actions
* Experiment tracking using MLflow
* Prediction logging for future monitoring and retraining

---

## Tech Stack

* Python
* Scikit-learn
* XGBoost
* Streamlit
* MLflow
* SHAP
* GitHub Actions

---

## Project Structure

```
FRAUD_DETECTION_PROJECT/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── splits/
│   │   ├── final_dataset.csv
│   │   ├── merged_test.csv
│   │   └── predictions.csv
│
├── logs/
│   └── predictions_log.csv
│
├── models/
│   ├── xgboost_model.pkl
│   ├── preprocessor.pkl
│   ├── selected_features.pkl
│   ├── feature_importance.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.txt
│   └── shap_plots/
│
├── notebooks/
│   └── model_selection.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── shap_explain.py
│   ├── split_all_data.py
│   ├── test_data_loader.py
│   └── sample_test.py
│
├── .github/workflows/
│   └── mlops.yaml
│
├── mlruns/
├── mlflow.db
├── requirements.txt
└── README.md
```

---

## MLOps Pipeline

```
Data Ingestion → Preprocessing → Model Training (MLflow)
       ↓
Model Saving → CI/CD (GitHub Actions)
       ↓
Deployment (Streamlit Dashboard)
       ↓
Prediction Logging → Future Monitoring and Retraining
```

---

## Installation

```
git clone <your-repository-link>
cd FRAUD_DETECTION_PROJECT
pip install -r requirements.txt
```

---

## Running the Application

```
streamlit run dashboard/app.py
```

---

## Model Selection

The following models were evaluated:

* Logistic Regression
* Random Forest
* XGBoost

Evaluation metric used: ROC-AUC score.

XGBoost was selected as the final model due to:

* Better performance on imbalanced data
* Higher ROC-AUC score
* Ability to capture complex non-linear patterns

---

## Explainability

SHAP (SHapley Additive exPlanations) is used to interpret model predictions and identify feature contributions. The dashboard provides visual explanations for flagged transactions.

---

## CI/CD Pipeline

The project uses GitHub Actions to automate:

* Model training
* Validation
* Pipeline execution on code updates

---

## Logging

Predictions are stored in:

```
logs/predictions_log.csv
```

This enables:

* Tracking model behavior
* Future retraining
* Extending to monitoring and drift detection

---

## Future Improvements

* Containerization using Docker
* Real-time inference using FastAPI
* Data drift detection
* Cloud deployment

---

## Author

Tushara

---

## Acknowledgements

IEEE-CIS Fraud Detection Dataset
Open-source machine learning community

---


## Dataset

Due to GitHub size limitations, datasets are not included in this repository.
You can download the dataset from:
https://www.kaggle.com/competitions/ieee-fraud-detection
