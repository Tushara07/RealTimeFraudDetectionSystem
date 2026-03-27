"""
train.py
--------
XGBoost training pipeline with full MLflow experiment tracking.

Usage
-----
    python src/train.py                         # default hyperparams
    python src/train.py --n-estimators 500      # override via CLI

MLflow UI
---------
    mlflow ui --port 5000
    open http://localhost:5000
"""

import os
import argparse
import logging
import pickle
import time

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

# local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_loader   import DataLoader
from preprocessing import Preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    n_estimators      = 300,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 10,
    scale_pos_weight  = 30,     # handles class imbalance (~1:30 in IEEE-CIS)
    eval_metric       = "aucpr",
    random_state      = 42,
    n_jobs            = -1,
    use_label_encoder = False,
)

MODELS_DIR    = "models"
MLFLOW_EXPMT  = "fraud-detection-xgboost"
CV_FOLDS      = 5


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_model(params: dict) -> XGBClassifier:
    return XGBClassifier(
        n_estimators      = params["n_estimators"],
        max_depth         = params["max_depth"],
        learning_rate     = params["learning_rate"],
        subsample         = params["subsample"],
        colsample_bytree  = params["colsample_bytree"],
        min_child_weight  = params["min_child_weight"],
        scale_pos_weight  = params["scale_pos_weight"],
        eval_metric       = params["eval_metric"],
        random_state      = params["random_state"],
        n_jobs            = params["n_jobs"],
        tree_method       = "hist",   # fast CPU training
        verbosity         = 0,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "roc_auc"           : roc_auc_score(y_true, y_prob),
        "avg_precision"     : average_precision_score(y_true, y_prob),
        "f1_score"          : f1_score(y_true, y_pred),
        "fraud_recall"      : classification_report(y_true, y_pred, output_dict=True)["1"]["recall"],
        "fraud_precision"   : classification_report(y_true, y_pred, output_dict=True)["1"]["precision"],
    }


def save_model(model: XGBClassifier, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", path)


# ── Main training function ─────────────────────────────────────────────────────

def train(params: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load & preprocess ──
    logger.info("── Step 1 / 4 : Loading data ──")
    loader = DataLoader()

    processed_path = os.path.join("data", "processed", "final_dataset.csv")
    if os.path.exists(processed_path):
        logger.info("Processed dataset found, loading directly.")
        df = loader.load_processed()
    else:
        logger.info("Raw data found, running full pipeline.")
        df = loader.run()

    logger.info("── Step 2 / 4 : Preprocessing ──")
    prep   = Preprocessor(models_dir=MODELS_DIR)
    X, y   = prep.fit_transform(df)
    prep.save()

    X_arr = X.values.astype(np.float32)
    y_arr = y.values

    # ── 2. MLflow run ──
    mlflow.set_experiment(MLFLOW_EXPMT)

    with mlflow.start_run(run_name=f"xgb_{int(time.time())}") as run:
        logger.info("MLflow run id: %s", run.info.run_id)

        # log hyperparams
        mlflow.log_params(params)
        mlflow.log_param("training_rows",    len(X_arr))
        mlflow.log_param("feature_count",    X_arr.shape[1])
        mlflow.log_param("fraud_rate_pct",   round(y_arr.mean() * 100, 3))

        # ── 3. Cross-validation ──
        logger.info("── Step 3 / 4 : Cross-validation (%d folds) ──", CV_FOLDS)
        model_cv = build_model(params)
        skf      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

        cv_roc = cross_val_score(model_cv, X_arr, y_arr, cv=skf, scoring="roc_auc", n_jobs=-1)
        cv_ap  = cross_val_score(model_cv, X_arr, y_arr, cv=skf, scoring="average_precision", n_jobs=-1)

        mlflow.log_metric("cv_roc_auc_mean",  cv_roc.mean())
        mlflow.log_metric("cv_roc_auc_std",   cv_roc.std())
        mlflow.log_metric("cv_avg_prec_mean", cv_ap.mean())
        mlflow.log_metric("cv_avg_prec_std",  cv_ap.std())

        logger.info("CV ROC-AUC : %.4f ± %.4f", cv_roc.mean(), cv_roc.std())
        logger.info("CV Avg-Prec: %.4f ± %.4f", cv_ap.mean(),  cv_ap.std())

        # ── 4. Final full-data fit ──
        logger.info("── Step 4 / 4 : Final full-data training ──")
        t0    = time.time()
        model = build_model(params)
        model.fit(X_arr, y_arr)
        train_time = time.time() - t0
        mlflow.log_metric("train_time_sec", round(train_time, 2))
        logger.info("Training finished in %.1f seconds.", train_time)

        # train-set metrics (sanity check, not eval metrics)
        y_prob = model.predict_proba(X_arr)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(y_arr, y_pred, y_prob)
        for name, val in metrics.items():
            mlflow.log_metric(f"train_{name}", round(val, 4))

        # confusion matrix as text artifact
        cm = confusion_matrix(y_arr, y_pred)
        cm_path = os.path.join(MODELS_DIR, "confusion_matrix.txt")
        np.savetxt(cm_path, cm, fmt="%d")
        mlflow.log_artifact(cm_path)

        # classification report
        report = classification_report(y_arr, y_pred, target_names=["legit", "fraud"])
        logger.info("\n%s", report)
        report_path = os.path.join(MODELS_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # log feature importances
        fi = pd.Series(
            model.feature_importances_,
            index=prep.selected_features
        ).sort_values(ascending=False)
        fi_path = os.path.join(MODELS_DIR, "feature_importance.csv")
        fi.to_csv(fi_path, header=["importance"])
        mlflow.log_artifact(fi_path)
        logger.info("Top-10 features:\n%s", fi.head(10))

        # save & log model
        model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
        save_model(model, model_path)
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

        # also save features list separately
        feat_path = os.path.join(MODELS_DIR, "selected_features.pkl")
        with open(feat_path, "wb") as f:
            pickle.dump(prep.selected_features, f)
        mlflow.log_artifact(feat_path)

        logger.info("✅  Run complete.  ROC-AUC=%.4f | Avg-Precision=%.4f",
                    metrics["roc_auc"], metrics["avg_precision"])
        print(f"\n🔗  MLflow run:  http://localhost:5000/#/experiments/... (run id: {run.info.run_id})")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost fraud detector")
    p.add_argument("--n-estimators",     type=int,   default=DEFAULTS["n_estimators"])
    p.add_argument("--max-depth",        type=int,   default=DEFAULTS["max_depth"])
    p.add_argument("--learning-rate",    type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--subsample",        type=float, default=DEFAULTS["subsample"])
    p.add_argument("--colsample-bytree", type=float, default=DEFAULTS["colsample_bytree"])
    p.add_argument("--min-child-weight", type=int,   default=DEFAULTS["min_child_weight"])
    p.add_argument("--scale-pos-weight", type=float, default=DEFAULTS["scale_pos_weight"])
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    params = {**DEFAULTS, **{
        k.replace("-", "_"): v for k, v in vars(args).items()
    }}
    train(params)