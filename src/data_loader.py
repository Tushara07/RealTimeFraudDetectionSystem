"""
data_loader.py
--------------
Loads and merges the IEEE-CIS Fraud Detection raw CSVs.
"""

import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Column groups ──────────────────────────────────────────────────────────────

# Transaction table: high-cardinality / memory-hungry columns we cast early
TRANSACTION_DTYPES: dict = {
    "TransactionID": "int32",
    "isFraud":       "int8",
    "TransactionDT": "int32",
    "TransactionAmt": "float32",
    # V-features (all float)
    **{f"V{i}": "float32" for i in range(1, 340)},
}

# Identity table
IDENTITY_DTYPES: dict = {
    "TransactionID": "int32",
}


class DataLoader:
    """
    Loads raw IEEE-CIS data and merges the two tables on TransactionID.

    Parameters
    ----------
    raw_dir : str
        Path to the folder containing train_transaction.csv and train_identity.csv.
    processed_dir : str
        Path where final_dataset.csv will be saved.
    """

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir       = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────────

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read both CSVs and return (transaction_df, identity_df)."""
        transaction_path = os.path.join(self.raw_dir, "train_transaction.csv")
        identity_path    = os.path.join(self.raw_dir, "train_identity.csv")

        logger.info("Loading transaction table from %s …", transaction_path)
        transaction_df = pd.read_csv(
            transaction_path,
            dtype=TRANSACTION_DTYPES,
            low_memory=False,
        )
        logger.info("Transaction table shape: %s", transaction_df.shape)

        logger.info("Loading identity table from %s …", identity_path)
        identity_df = pd.read_csv(
            identity_path,
            dtype=IDENTITY_DTYPES,
            low_memory=False,
        )
        logger.info("Identity table shape: %s", identity_df.shape)

        return transaction_df, identity_df

    def merge(
        self,
        transaction_df: pd.DataFrame,
        identity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left-join identity onto transaction on TransactionID.
        Rows without an identity record get NaN for identity columns.
        """
        logger.info("Merging tables …")
        merged = transaction_df.merge(identity_df, on="TransactionID", how="left")
        logger.info("Merged shape: %s  |  fraud rate: %.4f%%",
                    merged.shape,
                    merged["isFraud"].mean() * 100)
        return merged

    def save_processed(self, df: pd.DataFrame, filename: str = "final_dataset.csv") -> str:
        """Persist merged dataframe and return the output path."""
        out_path = os.path.join(self.processed_dir, filename)
        df.to_csv(out_path, index=False)
        logger.info("Saved processed dataset → %s", out_path)
        return out_path

    def load_processed(self, filename: str = "final_dataset.csv") -> pd.DataFrame:
        """Convenience loader for the already-merged CSV."""
        path = os.path.join(self.processed_dir, filename)
        logger.info("Loading processed dataset from %s …", path)
        df = pd.read_csv(path, low_memory=False)
        logger.info("Loaded shape: %s", df.shape)
        return df

    # ── one-shot helper ────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Load → merge → save → return the final dataframe in one call."""
        tx, ident = self.load_raw()
        merged    = self.merge(tx, ident)
        self.save_processed(merged)
        return merged


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.run()
    print(df.head())
    print("\nColumn count :", df.shape[1])
    print("Fraud samples :", df["isFraud"].sum())