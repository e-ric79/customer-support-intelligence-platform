"""
Cleans the Bitext customer support dataset and produces train/test CSVs.
Run this in Colab (or anywhere with HF access) — outputs land in training/data/.

Switched from the original 8k Kaggle-style dataset after diagnostics showed
its `Ticket Type` field was not actually derived from the ticket text
(model stuck at random-chance macro-F1 ~0.19 on 5 balanced classes, with a
uniform confusion matrix — the classic signature of an unlearnable label).

Bitext's `category` field is genuinely derived from the customer utterance
(it's an intent-classification dataset built for exactly this purpose), so
it should give the pipeline real signal to learn from.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

RANDOM_SEED = 42


def load_and_clean() -> pd.DataFrame:
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = ds["train"].to_pandas()

    # Keep only what we need: the customer utterance and its category.
    # No PII in this dataset (it's synthetic instruction/response pairs,
    # not real customer records) — no drop needed here, but always check
    # a new dataset for PII columns before trusting that.
    df = df[["instruction", "category"]].rename(
        columns={"instruction": "model_input", "category": "label"}
    )

    before = len(df)
    df = df.drop_duplicates(subset="model_input").reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows ({(before - len(df)) / before:.2%})")

    avg_words = df["model_input"].str.split().str.len().mean()
    print(f"Average model_input length: {avg_words:.1f} words (true word count)")

    return df


def main():
    df = load_and_clean()

    print("\nClass distribution:")
    print(df["label"].value_counts())

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["label"],  # keep class proportions intact in both splits
    )

    train_df.to_csv("training/data/train.csv", index=False)
    test_df.to_csv("training/data/test.csv", index=False)
    print(f"\nSaved {len(train_df)} train rows, {len(test_df)} test rows to training/data/")


if __name__ == "__main__":
    main()



def main():
    df = load_and_clean()

    print("\nClass distribution:")
    print(df["label"].value_counts())

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["label"],  # keep balance intact in both splits
    )

    train_df.to_csv("training/data/train.csv", index=False)
    test_df.to_csv("training/data/test.csv", index=False)
    print(f"\nSaved {len(train_df)} train rows, {len(test_df)} test rows to training/data/")


if __name__ == "__main__":
    main()
