import joblib
import streamlit as st
import os
import gdown

# Download model if not present
file_id = "1qcccO44Q7-zZqlWrlhEsCtmbLq0FzjVi"
model_path = "stack_model_new.pkl"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    st.write("Downloading model...")
    gdown.download(url, model_path, quiet=False)

# Load the model using joblib
try:
    model = joblib.load(model_path)
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Model could not be loaded. Error: {e}")
