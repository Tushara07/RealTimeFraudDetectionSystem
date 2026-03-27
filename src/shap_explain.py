"""
shap_explain.py
---------------
SHAP-based explainability for the XGBoost fraud detector.

Capabilities
------------
1. explain_one()         → SHAP values for a single transaction (used by dashboard)
2. explain_batch()       → SHAP values for a DataFrame
3. plot_waterfall()      → Waterfall plot  (single prediction)
4. plot_summary()        → Beeswarm / summary plot (dataset-level)
5. plot_bar()            → Mean |SHAP| bar chart (global feature importance)
6. get_top_features()    → Returns top-N feature contributions as a list of dicts

Usage
-----
    python src/shap_explain.py --input data/processed/final_dataset.csv --sample 500
"""

import os
import sys
import pickle
import logging
import argparse

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR        = "models"
MODEL_PATH        = os.path.join(MODELS_DIR, "xgboost_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
PLOTS_DIR         = os.path.join(MODELS_DIR, "shap_plots")

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import Preprocessor


# ── loader (cached) ────────────────────────────────────────────────────────────

_model        = None
_preprocessor = None
_explainer    = None


def _get_explainer():
    global _model, _preprocessor, _explainer

    if _explainer is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        _preprocessor = Preprocessor.load(PREPROCESSOR_PATH)

        logger.info("Building SHAP TreeExplainer …")
        _explainer = shap.TreeExplainer(
            _model,
            feature_perturbation="tree_path_dependent",
        )
        logger.info("Explainer ready.")

    return _explainer, _model, _preprocessor


# ── core SHAP computation ──────────────────────────────────────────────────────

def explain_batch(
    df: pd.DataFrame,
    sample_n: int = None,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """
    Compute SHAP values for a batch of raw transactions.

    Parameters
    ----------
    df       : Raw transaction DataFrame (same schema as training).
    sample_n : If set, randomly sample this many rows (speeds up large batches).

    Returns
    -------
    (shap_explanation, X_processed)
    """
    explainer, model, preprocessor = _get_explainer()

    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
        logger.info("Sampled %d rows for SHAP computation.", sample_n)

    X = preprocessor.transform(df.copy())
    X_arr = X.values.astype(np.float32)

    logger.info("Computing SHAP values for %d rows × %d features …", *X_arr.shape)
    shap_values = explainer(X_arr, check_additivity=False)

    # attach feature names
    shap_values.feature_names = preprocessor.selected_features
    return shap_values, X


def explain_one(
    record: dict,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """
    Compute SHAP values for a single transaction dict.

    Returns
    -------
    (shap_explanation_single_row, X_processed_single_row)
    """
    df = pd.DataFrame([record])
    shap_values, X = explain_batch(df)
    return shap_values[0], X


# ── plot helpers ───────────────────────────────────────────────────────────────

def _ensure_plots_dir() -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return PLOTS_DIR


def plot_waterfall(
    record: dict,
    max_display: int = 15,
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    Waterfall plot for a single transaction.
    Shows which features pushed the score up or down.

    Returns the saved image path.
    """
    shap_val, _ = explain_one(record)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.plots.waterfall(shap_val, max_display=max_display, show=False)
    plt.title("SHAP Waterfall — Single Transaction", fontsize=13, pad=12)
    plt.tight_layout()

    save_path = save_path or os.path.join(_ensure_plots_dir(), "waterfall.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    logger.info("Waterfall plot saved → %s", save_path)
    return save_path


def plot_summary(
    df: pd.DataFrame,
    sample_n: int = 1000,
    max_display: int = 20,
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    Beeswarm (summary) plot — global view of feature impact across many rows.
    """
    shap_values, X = explain_batch(df, sample_n=sample_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Summary — Feature Impact Distribution", fontsize=13, pad=12)
    plt.tight_layout()

    save_path = save_path or os.path.join(_ensure_plots_dir(), "summary_beeswarm.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    logger.info("Summary plot saved → %s", save_path)
    return save_path


def plot_bar(
    df: pd.DataFrame,
    sample_n: int = 1000,
    max_display: int = 20,
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    Global mean |SHAP| bar chart — overall feature importance.
    """
    shap_values, X = explain_batch(df, sample_n=sample_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    plt.sca(ax)
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title("Mean |SHAP| — Global Feature Importance", fontsize=13, pad=12)
    plt.tight_layout()

    save_path = save_path or os.path.join(_ensure_plots_dir(), "global_importance_bar.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    logger.info("Bar plot saved → %s", save_path)
    return save_path


# ── structured output (for dashboard / API) ────────────────────────────────────

def get_top_features(
    record: dict,
    top_n: int = 10,
) -> list[dict]:
    """
    Return top-N SHAP contributions for a single transaction as a list of dicts.

    Each dict has keys:
        feature   : str   — feature name
        value     : float — actual feature value in this transaction
        shap      : float — SHAP contribution (+ = pushed toward fraud)
        direction : str   — "increases_risk" | "decreases_risk"

    Sorted by abs(shap) descending.
    """
    shap_val, X = explain_one(record)

    explainer, model, preprocessor = _get_explainer()
    features = preprocessor.selected_features

    contributions = []
    for i, fname in enumerate(features):
        sv = float(shap_val.values[i])
        contributions.append({
            "feature"   : fname,
            "value"     : float(X.iloc[0][fname]),
            "shap"      : round(sv, 5),
            "direction" : "increases_risk" if sv > 0 else "decreases_risk",
        })

    contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return contributions[:top_n]


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SHAP plots for the fraud model.")
    p.add_argument("--input",   required=True,  help="CSV file to explain (raw format)")
    p.add_argument("--sample",  type=int, default=500,
                   help="Number of rows to sample for batch plots (default: 500)")
    p.add_argument("--plots",   nargs="+",
                   choices=["waterfall", "summary", "bar", "all"],
                   default=["all"],
                   help="Which plots to generate")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df   = pd.read_csv(args.input, low_memory=False)

    do_all      = "all" in args.plots
    do_waterfall = do_all or "waterfall" in args.plots
    do_summary   = do_all or "summary"   in args.plots
    do_bar       = do_all or "bar"       in args.plots

    if do_waterfall:
        sample_record = df.iloc[0].to_dict()
        path = plot_waterfall(sample_record, show=False)
        print(f"Waterfall plot  → {path}")

        top = get_top_features(sample_record, top_n=10)
        print("\nTop-10 SHAP contributions for first transaction:")
        print(f"  {'Feature':<15} {'Value':>10}  {'SHAP':>9}  Direction")
        print("  " + "-" * 55)
        for t in top:
            print(f"  {t['feature']:<15} {t['value']:>10.3f}  {t['shap']:>+9.5f}  {t['direction']}")

    if do_summary:
        path = plot_summary(df, sample_n=args.sample, show=False)
        print(f"\nSummary plot    → {path}")

    if do_bar:
        path = plot_bar(df, sample_n=args.sample, show=False)
        print(f"Bar chart       → {path}")