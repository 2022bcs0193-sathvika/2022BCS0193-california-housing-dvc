import pandas as pd

df = pd.read_csv("data/housing.csv")

# keep full dataset (no trimming)
df.to_csv("data/housing.csv", index=False)

print("Dataset Version 2 created with", len(df), "rows")
