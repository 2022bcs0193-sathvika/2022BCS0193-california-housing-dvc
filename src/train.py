import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json

ROLL_NO = "2022BCS0025"

df = pd.read_csv("data/housing.csv")

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Encode categorical column
df = pd.get_dummies(df, columns=["ocean_proximity"])

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

dataset_size = len(df)

# Detect dataset version
version = "Version 1" if dataset_size == 5000 else "Version 2"

print("===================================")
print(f"Roll No: {ROLL_NO}")
print(f"Dataset Version: {version}")
print(f"Dataset Size: {dataset_size}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
print("===================================")

metrics = {
    "roll_no": ROLL_NO,
    "version": version,
    "dataset_size": dataset_size,
    "rmse": rmse,
    "r2": r2
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)
