"""
Trains the baseline TF-IDF + Logistic Regression classifier.
Deliberately simple: the goal of v1 is a working end-to-end pipeline,
not maximum accuracy. Upgrade to a fine-tuned transformer later
(same reasoning as the Swahili sentiment project) once FastAPI/Docker/
K8s/CI-CD are proven with this lighter model.
"""

# import json
# import joblib
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline

# RANDOM_SEED = 42
# MODEL_PATH = "backend/model_artifacts/model.pkl"
# LABEL_MAP_PATH = "backend/model_artifacts/label_map.json"


# def main():
#     train_df = pd.read_csv("training/data/train.csv")

#     # unigrams + bigrams: bigrams matter here because phrases like
#     # "refund request" or "data loss" carry more signal than either
#     # word alone. max_features caps vocab size to keep the artifact small.
#     pipeline = Pipeline([
#         ("tfidf", TfidfVectorizer(
#             max_features=8000,
#             ngram_range=(1, 2),
#             stop_words="english",
#             min_df=2,
#         )),
#         ("clf", LogisticRegression(
#             max_iter=1000,
#             random_state=RANDOM_SEED,
#             class_weight="balanced",  # cheap insurance even though classes are ~balanced
#         )),
#     ])

#     pipeline.fit(train_df["model_input"], train_df["label"])

#     import os
#     os.makedirs("backend/model_artifacts", exist_ok=True)
#     joblib.dump(pipeline, MODEL_PATH)

#     label_map = sorted(train_df["label"].unique().tolist())
#     with open(LABEL_MAP_PATH, "w") as f:
#         json.dump(label_map, f, indent=2)

#     print(f"Saved model to {MODEL_PATH}")
#     print(f"Labels: {label_map}")


# if __name__ == "__main__":
#     main()

import json
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42

# ✅ MODIFIED: Define paths relative to the script's location (training/)
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(script_dir, "..", "backend")
model_dir = os.path.join(backend_dir, "model_artifacts")
MODEL_PATH = os.path.join(model_dir, "model.pkl")
LABEL_MAP_PATH = os.path.join(model_dir, "label_map.json")

# Data path relative to script
DATA_PATH = os.path.join(script_dir, "data", "train.csv")


def main():
    # Load data using absolute path
    train_df = pd.read_csv(DATA_PATH)

    # unigrams + bigrams: bigrams matter here because phrases like
    # "refund request" or "data loss" carry more signal than either
    # word alone. max_features caps vocab size to keep the artifact small.
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=8000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",  # cheap insurance even though classes are ~balanced
                ),
            ),
        ]
    )

    pipeline.fit(train_df["model_input"], train_df["label"])

    # ✅ MODIFIED: Ensure directory exists using absolute path
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    label_map = sorted(train_df["label"].unique().tolist())
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Labels: {label_map}")


if __name__ == "__main__":
    main()
