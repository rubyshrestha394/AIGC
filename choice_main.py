# choice_main.py
import streamlit as st
import nltk
from choice_inference import load_models, predict_text_single

# Download tokenizer data
nltk.download('punkt')

@st.cache_resource
def get_resources():
    return load_models()

# Load models & vectorizer
svm_model, rf_model, xgb_model, vectorizer = get_resources()

# Model selector mapping
model_options = {
    "SVM": svm_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# Streamlit UI
st.title("AI-Generated Content Detection")

input_text = st.text_area("Enter text to analyze:", height=200)

# Model selection dropdown
model_choice = st.selectbox("Choose the model for prediction:", list(model_options.keys()))

if st.button("Detect"):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        selected_model = model_options[model_choice]
        pred = predict_text_single(input_text, selected_model, vectorizer)

        label = "AI-Generated" if pred == 1 else "Human-Written"

        st.subheader(f"Prediction using **{model_choice}**:")
        st.markdown(f"### Result: **{label}**")
