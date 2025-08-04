# main.py
import streamlit as st
import nltk
from inference import load_models, predict_text

# Download tokenizer data (first run)
nltk.download('punkt')

@st.cache_resource
def get_resources():
    return load_models()  # Returns 4 values: svm, rf, xgb, vectorizer

# Load models and vectorizer
svm_model, rf_model, xgb_model, vectorizer = get_resources()
models = (svm_model, rf_model, xgb_model)

# Streamlit UI
st.title("AI-Generated Content Detection")

input_text = st.text_area("Enter text to analyze:", height=200)

if st.button("Detect"):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        preds = predict_text(input_text, models, vectorizer)
        def label(p): return "AI-Generated" if p == 1 else "Human-Written"

        st.subheader("Model Predictions:")
        st.write(f"**SVM**: {label(preds['svm'])}")
        st.write(f"**Random Forest**: {label(preds['rf'])}")
        st.write(f"**XGBoost**: {label(preds['xgb'])}")

        st.markdown("---")
        st.markdown(f"### Final Prediction: **{label(preds['vote'])}**")
