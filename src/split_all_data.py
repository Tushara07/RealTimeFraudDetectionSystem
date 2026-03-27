"""
split_all_data.py
-----------------
Splits both train (predictions.csv) and test datasets.
"""

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataSplitter:

    def __init__(self, n_splits: int = 4):
        self.n_splits = n_splits
        self.output_dir = "data/processed/splits"
        os.makedirs(self.output_dir, exist_ok=True)

    # ── TRAIN (STRATIFIED) ─────────────────────────────────────────────

    def split_train(self):
        path = "data/processed/predictions.csv"
        logger.info("Loading train data from %s …", path)

        df = pd.read_csv(path)

        if "isFraud" not in df.columns:
            raise ValueError("isFraud column missing")
        if "risk_tier" not in df.columns:
            raise ValueError("risk_tier column missing")

        df["strata"] = df["isFraud"].astype(str) + "_" + df["risk_tier"]

        splits = [[] for _ in range(self.n_splits)]

        for _, group in df.groupby("strata"):
            group = group.sample(frac=1, random_state=42)

            parts = np.array_split(group, self.n_splits)

            for i in range(self.n_splits):
                splits[i].append(parts[i])

        for i in range(self.n_splits):
            split_df = pd.concat(splits[i]).sample(frac=1, random_state=42)

            file_path = os.path.join(self.output_dir, f"train_part_{i+1}.csv")
            split_df.to_csv(file_path, index=False)

            logger.info("Saved %s with shape %s", file_path, split_df.shape)

    # ── TEST (RANDOM) ─────────────────────────────────────────────────

    def split_test(self):
        path = "data/processed/merged_test.csv"
        logger.info("Loading test data from %s …", path)

        df = pd.read_csv(path)

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        parts = np.array_split(df, self.n_splits)

        for i, part in enumerate(parts):
            file_path = os.path.join(self.output_dir, f"test_part_{i+1}.csv")
            part.to_csv(file_path, index=False)

            logger.info("Saved %s with shape %s", file_path, part.shape)

    # ── RUN ───────────────────────────────────────────────────────────

    def run(self):
        self.split_train()
        self.split_test()


if __name__ == "__main__":
    splitter = DataSplitter(n_splits=4)
    splitter.run()