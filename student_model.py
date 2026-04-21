import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv("datasets_11167_15520_train.csv")
print("✅ Dataset loaded successfully")
print(dataset.head())

# Features (X) and Target (y)
X = dataset.drop(["price_range"], axis=1)   # Drop target column
y = dataset["price_range"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train model
linear = linear_model.LogisticRegression(max_iter=200)
linear.fit(X_train, y_train)

# Accuracy
acc = linear.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {acc:.2f}")

# Save model with new version name
with open("cellphone_price_model_v2.pkl", "wb") as f:
    pickle.dump(linear, f)

print("📁 Model saved as cellphone_price_model_v2.pkl")
