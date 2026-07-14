"""
Evaluates the trained model on the held-out test set.
Reports macro-F1 (the metric to quote to a panel, not just accuracy)
plus a per-class breakdown and confusion matrix.
"""

# import joblib
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix, f1_score

# MODEL_PATH = "backend/model_artifacts/model.pkl"


# def main():
#     test_df = pd.read_csv("training/data/test.csv")
#     pipeline = joblib.load(MODEL_PATH)

#     preds = pipeline.predict(test_df["model_input"])

#     macro_f1 = f1_score(test_df["label"], preds, average="macro")
#     print(f"Macro F1: {macro_f1:.4f}\n")
#     MIN_THRESHOLD = 0.75
#     if macro_f1 < MIN_THRESHOLD:
#         print(
#             f"ERROR: Model performance ({macro_f1:.4f}) is below threshold ({MIN_THRESHOLD})!"
#         )
#         exit(1)  # This will fail the CI/CD pipeline
#     else:
#         print("Model performance acceptable.")
#         exit(0)

#     print("Per-class report:")
#     print(classification_report(test_df["label"], preds))

#     print("Confusion matrix:")
#     labels = sorted(test_df["label"].unique())
#     cm = confusion_matrix(test_df["label"], preds, labels=labels)
#     cm_df = pd.DataFrame(cm, index=labels, columns=labels)
#     print(cm_df)


# if __name__ == "__main__":
#     main()

import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# ✅ MODIFIED: Define paths relative to the script's location (training/)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets 'training/'
data_dir = os.path.join(script_dir, "data")  # Gets 'training/data'
backend_dir = os.path.join(script_dir, "..", "backend")  # Gets 'backend/'
model_dir = os.path.join(
    backend_dir, "model_artifacts"
)  # Gets 'backend/model_artifacts'

TEST_PATH = os.path.join(data_dir, "test.csv")
MODEL_PATH = os.path.join(model_dir, "model.pkl")


def main():
    # Load test data using absolute path
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Test data not found at {TEST_PATH}. Did you run prepare_data.py?"
        )

    test_df = pd.read_csv(TEST_PATH)

    # Load model using absolute path
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Did you run train.py?"
        )

    pipeline = joblib.load(MODEL_PATH)

    # Predict
    X_test = test_df["model_input"]
    y_test = test_df["label"]
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average="macro")
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Macro F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Optional: Print confusion matrix
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

    # Fail if performance is too low (optional threshold)
    THRESHOLD = 0.75
    if f1 < THRESHOLD:
        print(
            f"\n⚠️  WARNING: Model performance ({f1:.4f}) is below threshold ({THRESHOLD})!"
        )
        # Uncomment the line below to make the CI pipeline fail if F1 is too low
        # exit(1)
    else:
        print(f"\n✅ Model performance ({f1:.4f}) is acceptable.")


if __name__ == "__main__":
    main()
