from utils import load_jsonl
from evaluate import evaluate_model
from train import train_svm, train_rf, train_xgb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib
import os

def main():
    DATA_PATH = "../data_cleaning/train_dataset_with_features.jsonl"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = load_jsonl(DATA_PATH)

    # Vectorize cleaned text
    print("\nTfidfVectorizer...")
    vectorizer = TfidfVectorizer(max_features=5005, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df['cleaned_text'])

    # Save vectorizer
    print("\nSaving TfidfVectorizer...")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    # Extra numerical features
    extra_cols = ['text_length', 'avg_words_per_sentence', 'readability', 'punctuation_count', 'type_token_ratio']
    if not all(col in df.columns for col in extra_cols):
        raise ValueError(f"Missing one or more required extra features: {extra_cols}")

    X_extra = df[extra_cols].values
    X = hstack([X_tfidf, X_extra])
    y = df['label'].values

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Train & Evaluate SVM
    print("\nTraining SVM...")
    svm_model = train_svm(X_train, y_train)
    evaluate_model(svm_model, "SVM", X_val, y_val, X_test, y_test)
    joblib.dump(svm_model, "svm_model.pkl")

    # Train & Evaluate Random Forest
    print("\nTraining Random Forest...")
    rf_model = train_rf(X_train, y_train)
    evaluate_model(rf_model, "Random Forest", X_val, y_val, X_test, y_test)
    joblib.dump(rf_model, "rf_model.pkl")

    # Train & Evaluate XGBoost
    print("\nTraining XGBoost...")
    xgb_model = train_xgb(X_train, y_train)
    evaluate_model(xgb_model, "XGBoost", X_val, y_val, X_test, y_test)
    joblib.dump(xgb_model, "xgb_model.pkl")

    print("\nAll models trained, evaluated, and saved.")

if __name__ == "__main__":
    main()
