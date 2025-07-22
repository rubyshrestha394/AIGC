# inference.py
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
    ssvm_model  = joblib.load(os.path.join(model_dir, "svm_model.pkl"))
    rf_model   = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
    xgb_model  = joblib.load(os.path.join(model_dir, "xgb_model.pkl"))
    vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    return svm_model, rf_model, xgb_model, vectorizer


def prepare_features(text: str, vectorizer, expected_dim: int):
    """
    Transform raw text into feature vector matching the model's expected input dimension.

    Args:
        text (str): Input document text.
        vectorizer: Fitted TF-IDF vectorizer.
        expected_dim (int): Number of features the model expects.

    Returns:
        scipy.sparse.csr_matrix: Feature matrix ready for prediction.
    """
    tfidf_vec = vectorizer.transform([text])
    actual_dim = tfidf_vec.shape[1]
    if actual_dim < expected_dim:
        pad_width = expected_dim - actual_dim
        zero_pad  = csr_matrix(np.zeros((1, pad_width)))
        features  = hstack([tfidf_vec, zero_pad])
    else:
        # Truncate if vectorizer produces more features
        features = tfidf_vec[:, :expected_dim]
    return features


def predict_text(text: str, models, vectorizer):
    """
    Run SVM, Random Forest, and XGBoost predictions on input text.

    Args:
        text (str): Raw input text to classify.
        models (tuple): (svm_model, rf_model, xgb_model).
        vectorizer: TF-IDF vectorizer.

    Returns:
        dict: Predictions from each model and final vote.
    """
    svm_model, rf_model, xgb_model = models
    expected_dim = svm_model.n_features_in_
    X_input = prepare_features(text, vectorizer, expected_dim)

    preds = {
        'svm':  svm_model.predict(X_input)[0],
        'rf':   rf_model.predict(X_input)[0],
        'xgb':  xgb_model.predict(X_input)[0]
    }

    # Majority vote (simple average + round)
    vote = int(np.round(np.mean(list(preds.values()))))
    preds['vote'] = vote
    return preds
