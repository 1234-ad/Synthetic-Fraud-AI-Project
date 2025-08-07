# generate_creditcard.py

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Create 1000 rows of synthetic anonymized data
num_samples = 1000
feature_columns = [f"V{i}" for i in range(1, 29)]

# Simulate features with Gaussian distribution
X = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, len(feature_columns)))
df = pd.DataFrame(X, columns=feature_columns)

# Add 'Amount' and 'Time' features with skew
df["Amount"] = np.abs(np.random.exponential(scale=50, size=num_samples))
df["Time"] = np.random.randint(0, 172800, size=num_samples)  # two days in seconds

# Add highly imbalanced 'Class' target
df["Class"] = np.random.choice([0, 1], size=num_samples, p=[0.985, 0.015])

# Save to CSV
df.to_csv("Data/creditcard.csv", index=False)

print("✅ creditcard.csv generated successfully!")
