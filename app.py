import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# -------------------------------
# File download / load logic
# -------------------------------
MODEL_ID = "1qcccO44Q7-zZqlWrlhEsCtmbLq0FzjVi"
MODEL_PATH = "stack_model_new.pkl"
ENCODER_PATH = "label_encoder_new.pkl"
SCALER_PATH = "scaler_new.pkl"
IMPUTER_PATH = "imputer.pkl"

def download_file(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        st.info(f"Downloading {output_path} ...")
        gdown.download(url, output_path, quiet=True)
    # else:
    #     st.write(f"{output_path} already exists.")

# Download all required files if missing
download_file(MODEL_ID, MODEL_PATH)
# Add download_file calls for ENCODER_PATH, SCALER_PATH, IMPUTER_PATH if you have them on Drive

# -------------------------------
# Load saved objects
# -------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, encoder, scaler, imputer

model, le, scaler, imputer = load_assets()

# -------------------------------
# Streamlit setup
# -------------------------------
st.set_page_config(page_title="Music Genre Classification", page_icon="🎵", layout="wide")
st.title("🎶 Music Genre Classification using Stacking Ensemble")
st.write(
    "Predict the **music genre** based on song features using a Stacking Ensemble "
    "of XGBoost, LightGBM, and CatBoost models."
)

# Sidebar
option = st.sidebar.radio("Choose input method:", ["Manual Input", "Upload CSV"])

# -------------------------------
# Feature list
# -------------------------------
features = [
    'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
    'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature', 'log_duration',
    'energy_danceability', 'speech_acoustic', 'energy_to_loudness',
    'acoustic_to_instrumental', 'tempo_bin'
]

# -------------------------------
# Helper: Auto Feature Engineering
# -------------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['log_duration'] = np.log1p(df['duration_ms']) if 'duration_ms' in df.columns else 0
    df['energy_danceability'] = df.get('energy', 0) * df.get('danceability', 0)
    df['speech_acoustic'] = df.get('speechiness', 0) * df.get('acousticness', 0)
    df['energy_to_loudness'] = df.get('energy', 0) / (df.get('loudness', 1) + 1e-6)
    df['acoustic_to_instrumental'] = df.get('acousticness', 0) / (df.get('instrumentalness', 1) + 1e-6)
    df['tempo_bin'] = pd.cut(df['tempo'], bins=[0, 90, 120, 150, 200, np.inf], labels=[0, 1, 2, 3, 4]).astype(float) if 'tempo' in df.columns else 0.0
    return df

# -------------------------------
# Manual Input
# -------------------------------
if option == "Manual Input":
    st.subheader("Enter Song Features")
    user_data = {f: st.number_input(f"{f}", value=0.0) for f in [
        'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature'
    ]}
    input_df = pd.DataFrame([user_data])
    input_df = add_derived_features(input_df)

# -------------------------------
# CSV Upload
# -------------------------------
else:
    st.subheader("Upload a CSV file")
    uploaded_file = st.file_uploader("Upload your CSV with base feature columns", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(input_df.head())
        input_df = add_derived_features(input_df)
    else:
        input_df = None

# -------------------------------
# Prediction
# -------------------------------
if st.button("🎧 Predict Genre"):
    if input_df is not None:
        # Impute and scale
        try:
            input_df = imputer.transform(input_df)
            input_df = scaler.transform(input_df)
        except Exception:
            pass

        # Predict
        preds = model.predict(input_df)

        # Decode
        try:
            decoded_preds = le.inverse_transform(preds)
        except Exception:
            preds = preds.astype(int)
            decoded_preds = le.inverse_transform(preds)

        # Display results
        if len(decoded_preds) == 1:
            st.success(f"🎵 Predicted Genre: **{decoded_preds[0]}**")
        else:
            result_df = pd.DataFrame({"Predicted Genre": decoded_preds})
            st.write("### 🎶 Predicted Genres for Uploaded Songs:")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Predictions as CSV",
                csv,
                "predicted_genres.csv",
                "text/csv"
            )
    else:
        st.warning("Please provide input data first!")
