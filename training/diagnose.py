"""
Diagnostic: is there any learnable signal between model_input and label?
Run this before assuming the pipeline code is broken.
"""
import joblib
import pandas as pd
from sklearn.metrics import f1_score

MODEL_PATH = "backend/model_artifacts/model.pkl"

train_df = pd.read_csv("training/data/train.csv")
pipeline = joblib.load(MODEL_PATH)

# 1. Can the model even memorize its OWN training data?
train_preds = pipeline.predict(train_df["model_input"])
train_f1 = f1_score(train_df["label"], train_preds, average="macro")
print(f"TRAIN macro-F1 (memorization check): {train_f1:.4f}")
print("If this is also ~0.20, there is no learnable signal at all — not a generalization problem.\n")

# 2. Look at raw text side-by-side across labels — does it read differently by eye?
print("Sample rows per label:")
for label in sorted(train_df["label"].unique()):
    sample = train_df[train_df["label"] == label]["model_input"].iloc[0]
    print(f"\n[{label}]\n{sample[:300]}")
