# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os

# Load dataset
df = pd.read_csv("Data/augmented_dataset.csv")

# Features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train model
clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("🎯 ROC AUC Score:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

# Save model
os.makedirs("Model", exist_ok=True)
with open("Model/fraud_model_rf.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Model saved as fraud_model_rf.pkl")
