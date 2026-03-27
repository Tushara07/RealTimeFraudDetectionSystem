import pandas as pd

df = pd.read_csv("data/processed/merged_test.csv")

# Take first 10,000 rows (or random sample)
df_small = df.sample(n=10000, random_state=42)

df_small.to_csv("data/processed/merged_test_small.csv", index=False)

print("Saved smaller file")