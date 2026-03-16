import pandas as pd

df = pd.read_csv("data/housing.csv")

# keep only first 5000 rows
df = df.head(5000)

df.to_csv("data/housing.csv", index=False)

print("Dataset Version 1 created with", len(df), "rows")
