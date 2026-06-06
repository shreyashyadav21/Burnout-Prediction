import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("employee_burnout_dataset_1000_records.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Remove employee_id
df = df.drop("employee_id", axis=1)

# Features and target
X = df.drop("burnout_risk_score", axis=1).values
y = df["burnout_risk_score"].values

# Train-Test Split (80-20)
split_index = int(0.8 * len(X))

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# Add bias column
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Linear Regression using Normal Equation
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Predictions
y_pred = X_test @ theta

# Evaluation Metrics

mae = np.mean(np.abs(y_test - y_pred))

mse = np.mean((y_test - y_pred) ** 2)

rmse = np.sqrt(mse)

r2 = 1 - (
    np.sum((y_test - y_pred) ** 2)
    / np.sum((y_test - np.mean(y_test)) ** 2)
)

print("\nResults")
print("MAE =", round(mae, 4))
print("MSE =", round(mse, 4))
print("RMSE =", round(rmse, 4))
print("R2 Score =", round(r2, 4))

# Coefficients
feature_names = ["Intercept"] + list(df.drop("burnout_risk_score", axis=1).columns)

print("\nFeature Coefficients")

for name, coef in zip(feature_names, theta):
    print(f"{name}: {coef:.4f}")