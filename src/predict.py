"""
predict.py
----------
Inference module for the trained XGBoost fraud detector.

Supports
--------
1. Single transaction  → predict_one(record: dict)
2. Batch (DataFrame)   → predict_batch(df: pd.DataFrame)
3. CSV file            → predict_csv(input_path, output_path)
4. CLI                 → python src/predict.py --input data/raw/test.csv

Output columns added to input
------------------------------
fraud_probability   float  [0, 1]
fraud_predicted     int    {0, 1}  (threshold default = 0.5)
risk_tier           str    Low / Medium / High / Critical
"""

import os
import sys
import pickle
import logging
import argparse
from typing import Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── paths (relative to project root) ──────────────────────────────────────────
MODELS_DIR      = "models"
MODEL_PATH      = os.path.join(MODELS_DIR, "xgboost_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
FEATURES_PATH   = os.path.join(MODELS_DIR, "selected_features.pkl")

# ── risk tier thresholds ───────────────────────────────────────────────────────
RISK_TIERS = [
    (0.80, "Critical"),
    (0.50, "High"),
    (0.20, "Medium"),
    (0.00, "Low"),
]

DEFAULT_THRESHOLD = 0.5


# ── helpers ────────────────────────────────────────────────────────────────────

def _risk_tier(prob: float) -> str:
    for cutoff, label in RISK_TIERS:
        if prob >= cutoff:
            return label
    return "Low"


def _load_artifacts() -> tuple:
    """Load model + preprocessor from disk. Cached after first call."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run src/train.py first."
        )
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found at '{PREPROCESSOR_PATH}'. Run src/train.py first."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", MODEL_PATH)

    # import here to avoid circular imports when used as a module
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import Preprocessor
    preprocessor = Preprocessor.load(PREPROCESSOR_PATH)

    return model, preprocessor


# ── module-level cache (avoids reloading on repeated calls) ───────────────────
_model        = None
_preprocessor = None


def _get_artifacts():
    global _model, _preprocessor
    if _model is None or _preprocessor is None:
        _model, _preprocessor = _load_artifacts()
    return _model, _preprocessor


# ── public API ─────────────────────────────────────────────────────────────────

def predict_batch(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Run inference on a DataFrame of raw transactions.

    Parameters
    ----------
    df        : Raw transaction data (same schema as training CSVs).
    threshold : Decision threshold for fraud_predicted label.

    Returns
    -------
    Original DataFrame with three new columns appended.
    """
    model, preprocessor = _get_artifacts()

    logger.info("Preprocessing %d rows …", len(df))
    X = preprocessor.transform(df.copy())
    X_arr = X.values.astype(np.float32)

    logger.info("Running inference …")
    probs = model.predict_proba(X_arr)[:, 1]
    preds = (probs >= threshold).astype(int)
    tiers = [_risk_tier(p) for p in probs]

    result = df.copy()
    result["fraud_probability"] = np.round(probs, 4)
    result["fraud_predicted"]   = preds
    result["risk_tier"]         = tiers

    n_fraud = preds.sum()
    logger.info(
        "Done. Flagged %d / %d transactions as fraud (%.2f%%)",
        n_fraud, len(df), n_fraud / len(df) * 100,
    )
    return result


def predict_one(
    record: dict,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Score a single transaction dict.

    Parameters
    ----------
    record : dict with transaction fields (missing fields → NaN).

    Returns
    -------
    dict with keys: fraud_probability, fraud_predicted, risk_tier
    """
    df     = pd.DataFrame([record])
    result = predict_batch(df, threshold=threshold)
    row    = result.iloc[0]
    return {
        "fraud_probability": float(row["fraud_probability"]),
        "fraud_predicted":   int(row["fraud_predicted"]),
        "risk_tier":         row["risk_tier"],
    }


def predict_csv(
    input_path: str,
    output_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    identity_path: str = None,
) -> str:
    """
    Read a CSV (or two CSVs), score it, write results CSV.

    For the IEEE-CIS test set, pass both files:
        input_path    = data/raw/test_transaction.csv
        identity_path = data/raw/test_identity.csv   (optional)

    Returns the output path.
    """
    logger.info("Loading input CSV: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)

    if identity_path and os.path.exists(identity_path):
        logger.info("Merging identity file: %s", identity_path)
        id_df = pd.read_csv(identity_path, low_memory=False)
        df = df.merge(id_df, on="TransactionID", how="left")
        logger.info("Merged shape: %s", df.shape)

    result = predict_batch(df, threshold=threshold)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Predictions saved → %s", output_path)
    return output_path


# ── summary report ─────────────────────────────────────────────────────────────

def prediction_summary(result_df: pd.DataFrame) -> None:
    """Print a quick summary of a scored DataFrame."""
    total = len(result_df)
    fraud = result_df["fraud_predicted"].sum()

    print("\n" + "=" * 45)
    print("  FRAUD DETECTION — PREDICTION SUMMARY")
    print("=" * 45)
    print(f"  Total transactions : {total:,}")
    print(f"  Flagged as fraud   : {fraud:,}  ({fraud/total*100:.2f}%)")
    print("\n  Risk Tier Breakdown:")
    tier_counts = result_df["risk_tier"].value_counts()
    for tier in ["Critical", "High", "Medium", "Low"]:
        count = tier_counts.get(tier, 0)
        bar   = "█" * int(count / total * 40)
        print(f"    {tier:<10} {count:>6,}  {bar}")
    print("=" * 45)

    print("\n  Top 10 highest-risk transactions:")
    cols = ["TransactionID", "TransactionAmt", "fraud_probability", "risk_tier"] \
        if "TransactionID" in result_df.columns else \
        ["fraud_probability", "risk_tier"]
    print(
        result_df.nlargest(10, "fraud_probability")[cols].to_string(index=False)
    )
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fraud predictions on a CSV file.")
    p.add_argument("--input",     required=True,  help="Path to transaction CSV (e.g. test_transaction.csv)")
    p.add_argument("--identity",  default=None,   help="Path to identity CSV (e.g. test_identity.csv) — optional")
    p.add_argument("--output",    default="data/processed/predictions.csv",
                                  help="Path for output CSV (default: data/processed/predictions.csv)")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                                  help="Decision threshold (default: 0.5)")
    p.add_argument("--summary",   action="store_true",
                                  help="Print summary report after scoring")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    result = predict_csv(args.input, args.output, threshold=args.threshold, identity_path=args.identity)

    if args.summary:
        df_result = pd.read_csv(args.output)
        prediction_summary(df_result)