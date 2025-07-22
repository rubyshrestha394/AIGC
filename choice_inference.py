# choice_inference.py
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
import os

def load_models(model_dir: str = "models"):
    """
    Load trained models and the TF-IDF vectorizer from disk.

    Returns:
        tuple: (svm_model, rf_model, xgb_model, vectorizer)
    """
    svm_model  = joblib.load(os.path.join(model_dir, "svm_model.pkl"))
    rf_model   = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
    xgb_model  = joblib.load(os.path.join(model_dir, "xgb_model.pkl"))
    vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    return svm_model, rf_model, xgb_model, vectorizer


def prepare_features(text: str, vectorizer, expected_dim: int):
    """
    Transform raw text into feature vector matching the model's expected input dimension.
    """
    tfidf_vec = vectorizer.transform([text])
    actual_dim = tfidf_vec.shape[1]
    if actual_dim < expected_dim:
        pad_width = expected_dim - actual_dim
        zero_pad = csr_matrix(np.zeros((1, pad_width)))
        features = hstack([tfidf_vec, zero_pad])
    else:
        features = tfidf_vec[:, :expected_dim]
    return features


def predict_text_single(text: str, model, vectorizer):
    """
    Predict using a single model (SVM, RF, or XGB).

    Args:
        text (str): Input text.
        model: Trained classifier (must have n_features_in_).
        vectorizer: TF-IDF vectorizer.

    Returns:
        int: 1 for AI-generated, 0 for human-written.
    """
    expected_dim = model.n_features_in_
    X_input = prepare_features(text, vectorizer, expected_dim)
    return model.predict(X_input)[0]
