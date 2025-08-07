# generate_synthetic_fraud.py

import pandas as pd

# Load the real dataset
df = pd.read_csv("Data/creditcard.csv")

# Filter for fraud cases only (Class = 1)
fraud_cases = df[df["Class"] == 1]

# Oversample fraud samples to generate synthetic fraud-like data
synthetic_fraud = fraud_cases.sample(n=500, replace=True, random_state=42).copy()

# Save as new synthetic file
synthetic_fraud.to_csv("Data/synthetic_fraud.csv", index=False)

print("✅ synthetic_fraud.csv generated from oversampled real fraud data.")
