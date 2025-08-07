# create_augmented_dataset.py

import pandas as pd

# Load both real and synthetic fraud data
real_data = pd.read_csv("Data/creditcard.csv")
synthetic_fraud = pd.read_csv("Data/synthetic_fraud.csv")

# Concatenate both datasets
augmented_data = pd.concat([real_data, synthetic_fraud], ignore_index=True)

# Shuffle the rows randomly
augmented_data = augmented_data.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Save final dataset
augmented_data.to_csv("Data/augmented_dataset.csv", index=False)

print("✅ augmented_dataset.csv created with real + synthetic data.")
