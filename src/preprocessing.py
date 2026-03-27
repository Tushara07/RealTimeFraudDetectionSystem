"""
preprocessing.py
----------------
Feature engineering + preprocessing for the IEEE-CIS fraud dataset.

Steps
-----
1. Drop high-missing-rate columns
2. Separate numeric / categorical columns
3. Impute missing values
4. Encode categoricals (label encoding – compatible with XGBoost)
5. (Optional) SMOTE oversampling for class imbalance
6. Return X, y + the list of selected features
"""

import os
import pickle
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# Columns that are IDs / leakage / not useful as features
DROP_COLS = ["TransactionID", "TransactionDT"]

# Fraction of missing values above which a column is dropped entirely
MISSING_THRESHOLD = 0.5


class Preprocessor:
    """
    Full preprocessing pipeline for IEEE-CIS fraud data.

    Parameters
    ----------
    missing_threshold : float
        Drop columns whose missing-value ratio exceeds this value.
    models_dir : str
        Directory to save selected_features.pkl.
    """

    def __init__(
        self,
        missing_threshold: float = MISSING_THRESHOLD,
        models_dir: str = "models",
    ):
        self.missing_threshold = missing_threshold
        self.models_dir        = models_dir
        os.makedirs(models_dir, exist_ok=True)

        self.num_imputer  = SimpleImputer(strategy="median")
        self.cat_imputer  = SimpleImputer(strategy="most_frequent")
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.selected_features: list[str] = []
        self._fitted = False

    # ── private helpers ────────────────────────────────────────────────────────

    def _drop_high_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_rate = df.isnull().mean()
        keep = missing_rate[missing_rate <= self.missing_threshold].index.tolist()
        dropped = [c for c in df.columns if c not in keep]
        if dropped:
            logger.info("Dropping %d high-missing columns: %s …", len(dropped), dropped[:5])
        return df[keep]

    def _split_feature_types(
        self, df: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        return num_cols, cat_cols

    # ── public API ─────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "isFraud",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fit on training data and return (X, y).
        Call this once during training.
        """
        logger.info("Starting fit_transform …")

        # ── target ──
        y = df[target_col].astype(int)
        df = df.drop(columns=[target_col] + [c for c in DROP_COLS if c in df.columns])

        # ── drop high-missing ──
        df = self._drop_high_missing(df)

        num_cols, cat_cols = self._split_feature_types(df)
        logger.info("Numeric: %d  |  Categorical: %d", len(num_cols), len(cat_cols))

        # ── impute numerics ──
        if num_cols:
            df[num_cols] = self.num_imputer.fit_transform(df[num_cols])

        # ── impute + label-encode categoricals ──
        if cat_cols:
            df[cat_cols] = self.cat_imputer.fit_transform(df[cat_cols])
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # store fitted column lists so transform() uses the exact same split
        self._fitted_num_cols = num_cols
        self._fitted_cat_cols = cat_cols
        self.selected_features = df.columns.tolist()
        self._fitted = True

        self._save_features()
        logger.info("fit_transform complete. Feature count: %d", len(self.selected_features))
        return df, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to new data (inference time).
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before calling transform().")

        # drop non-feature cols if present
        drop = [c for c in DROP_COLS + ["isFraud"] if c in df.columns]
        df = df.drop(columns=drop)

        # keep only trained features (add missing ones as NaN)
        for col in self.selected_features:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.selected_features]

        # use the exact same column split from fit — never re-detect from dtypes
        num_cols = [c for c in getattr(self, "_fitted_num_cols", []) if c in df.columns]
        cat_cols = [c for c in getattr(self, "_fitted_cat_cols", []) if c in df.columns]

        if num_cols:
            df[num_cols] = self.num_imputer.transform(df[num_cols])

        if cat_cols:
            df[cat_cols] = self.cat_imputer.transform(df[cat_cols])
            for col in cat_cols:
                le = self.label_encoders.get(col)
                if le:
                    # handle unseen labels → map to most-frequent class
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])

        return df

    # ── persistence helpers ────────────────────────────────────────────────────

    def _save_features(self) -> None:
        path = os.path.join(self.models_dir, "selected_features.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.selected_features, f)
        logger.info("Saved selected features → %s", path)

    def save(self, path: Optional[str] = None) -> str:
        """Pickle the entire Preprocessor (imputers + encoders + features)."""
        path = path or os.path.join(self.models_dir, "preprocessor.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Preprocessor saved → %s", path)
        return path

    @staticmethod
    def load(path: str) -> "Preprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Preprocessor loaded from %s", path)
        return obj


# ── standalone smoke-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import DataLoader

    df     = DataLoader().run()
    prep   = Preprocessor()
    X, y   = prep.fit_transform(df)

    print("X shape :", X.shape)
    print("y dist  :\n", y.value_counts())
    prep.save()