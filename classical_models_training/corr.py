import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from utils import load_jsonl


def compute_and_show_correlation(df, extra_cols, X_tfidf, target_col='label'):
    """
    Compute and display the correlation matrix for the specified extra columns, target, and top TF-IDF features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        extra_cols (list of str): Names of the numerical feature columns.
        X_tfidf (sparse matrix): TF-IDF feature matrix.
        target_col (str): Name of the target label column.
    """
    from sklearn.feature_selection import VarianceThreshold

    # Convert sparse matrix to dense for correlation (sample top 20 high-variance features)
    selector = VarianceThreshold(threshold=0.0001)
    X_reduced = selector.fit_transform(X_tfidf)

    tfidf_top = pd.DataFrame(X_reduced[:, :20].toarray(), columns=[f'tfidf_{i}' for i in range(20)])
    features_df = pd.concat([df[extra_cols + [target_col]].reset_index(drop=True), tfidf_top], axis=1)

    corr_df = features_df.corr()
    print("\nCorrelation matrix:")
    print(corr_df)

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix of Extra Features + Top TF-IDF + Label")
    plt.tight_layout()
    plt.show()


def main():
    DATA_PATH = os.path.join("..", "data_cleaning", "train_dataset_with_features.jsonl")
    TFIDF_PATH = "../models/tfidf_vectorizer.pkl"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = load_jsonl(DATA_PATH)
    print("Loaded DataFrame with shape:", df.shape)

    # Define the extra numerical feature columns
    extra_cols = [
        'text_length',
        'avg_words_per_sentence',
        'readability',
        'punctuation_count',
        'type_token_ratio'
    ]

    # Verify columns exist
    missing = [c for c in extra_cols + ['label'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Load or fit TF-IDF
    if os.path.exists(TFIDF_PATH):
        print("\nLoading existing TfidfVectorizer...")
        vectorizer = joblib.load(TFIDF_PATH)
        X_tfidf = vectorizer.transform(df['cleaned_text'])
    else:
        print("\nFitting new TfidfVectorizer...")
        vectorizer = TfidfVectorizer(max_features=5005, ngram_range=(1, 2))
        X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
        print("\nSaving TfidfVectorizer...")
        joblib.dump(vectorizer, TFIDF_PATH)

    # Compute and show correlation
    compute_and_show_correlation(df, extra_cols, X_tfidf, target_col='label')


if __name__ == "__main__":
    main()