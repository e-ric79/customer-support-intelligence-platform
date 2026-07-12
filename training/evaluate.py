"""
Evaluates the trained model on the held-out test set.
Reports macro-F1 (the metric to quote to a panel, not just accuracy)
plus a per-class breakdown and confusion matrix.
"""
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

MODEL_PATH = "backend/model_artifacts/model.pkl"


def main():
    test_df = pd.read_csv("training/data/test.csv")
    pipeline = joblib.load(MODEL_PATH)

    preds = pipeline.predict(test_df["model_input"])

    macro_f1 = f1_score(test_df["label"], preds, average="macro")
    print(f"Macro F1: {macro_f1:.4f}\n")

    print("Per-class report:")
    print(classification_report(test_df["label"], preds))

    print("Confusion matrix:")
    labels = sorted(test_df["label"].unique())
    cm = confusion_matrix(test_df["label"], preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)


if __name__ == "__main__":
    main()
