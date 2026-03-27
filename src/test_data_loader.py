"""
test_data_loader.py
------------------
Loads and merges the IEEE-CIS test CSVs.
"""

import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Column groups ──────────────────────────────────────────────────────────────

TRANSACTION_DTYPES: dict = {
    "TransactionID": "int32",
    "TransactionDT": "int32",
    "TransactionAmt": "float32",
    **{f"V{i}": "float32" for i in range(1, 340)},
}

IDENTITY_DTYPES: dict = {
    "TransactionID": "int32",
}


class TestDataLoader:
    """
    Loads raw IEEE-CIS test data and merges transaction + identity tables.

    Parameters
    ----------
    raw_dir : str
        Path to folder containing test_transaction.csv and test_identity.csv
    processed_dir : str
        Path where merged_test.csv will be saved
    """

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────────

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load test_transaction and test_identity CSVs."""
        transaction_path = os.path.join(self.raw_dir, "test_transaction.csv")
        identity_path    = os.path.join(self.raw_dir, "test_identity.csv")

        logger.info("Loading test transaction data from %s …", transaction_path)
        transaction_df = pd.read_csv(
            transaction_path,
            dtype=TRANSACTION_DTYPES,
            low_memory=False,
        )
        logger.info("Transaction shape: %s", transaction_df.shape)

        logger.info("Loading test identity data from %s …", identity_path)
        identity_df = pd.read_csv(
            identity_path,
            dtype=IDENTITY_DTYPES,
            low_memory=False,
        )
        logger.info("Identity shape: %s", identity_df.shape)

        return transaction_df, identity_df

    def merge(
        self,
        transaction_df: pd.DataFrame,
        identity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge identity onto transaction using left join on TransactionID.
        Keeps all transactions and attaches identity where available.
        """
        logger.info("Merging test datasets …")
        merged = transaction_df.merge(identity_df, on="TransactionID", how="left")
        logger.info("Merged shape: %s", merged.shape)
        return merged

    def save_processed(self, df: pd.DataFrame, filename: str = "merged_test.csv") -> str:
        """Save merged dataset and return file path."""
        output_path = os.path.join(self.processed_dir, filename)
        df.to_csv(output_path, index=False)
        logger.info("Saved merged dataset → %s", output_path)
        return output_path

    def load_processed(self, filename: str = "merged_test.csv") -> pd.DataFrame:
        """Load already merged dataset."""
        path = os.path.join(self.processed_dir, filename)
        logger.info("Loading merged dataset from %s …", path)
        df = pd.read_csv(path, low_memory=False)
        logger.info("Loaded shape: %s", df.shape)
        return df

    # ── one-shot helper ────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Load → merge → save → return dataframe."""
        tx, ident = self.load_raw()
        merged = self.merge(tx, ident)
        self.save_processed(merged)
        return merged


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = TestDataLoader()
    df = loader.run()

    print(df.head())
    print("\nColumn count :", df.shape[1])
    print("Total rows   :", df.shape[0])