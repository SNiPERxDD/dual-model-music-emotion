# --- START OF FILE app.py ---

#USAGE : streamlit run app.py

import os
import numpy as np
import pandas as pd
import librosa
import joblib
import logging
import streamlit as st
import tempfile
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment

# --- Configuration ---
# Rate-limit the backend processing to prevent CPU/memory spikes.
# This does NOT affect the upload, only the analysis phase.
MAX_CONCURRENT_JOBS = 3

# --- Load Pre-trained Models ---
@st.cache_resource
def load_models():
    """Loads all the models and encoders into memory once."""
    try:
        deep_model = joblib.load("models/model.pkl")
        xgb_model = joblib.load("models/mood_xgboost_tuned.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        return deep_model, xgb_model, label_encoder
    except FileNotFoundError as e:
        st.error(f"Model loading failed: {e}. Ensure model files are in the 'models/' directory.")
        st.stop()

models = load_models()
if models:
    deep_model, xgb_model, label_encoder = models
    scaler = deep_model["scaler"]
    encoder = deep_model["encoder"]
    umap_model = deep_model["umap"]
    kmeans = deep_model["kmeans"]
    cluster_emotions = deep_model["cluster_emotions"]

# --- Feature Extraction & Prediction (These functions are stable) ---
def extract_features(file_path):
    try:
        # Using a fixed 30-second clip from 45 seconds in, as in the original code
        y, sr = librosa.load(file_path, offset=45.0, duration=30.0, mono=True)
        
        # If the clip is too short, librosa might return an empty array
        if len(y) < 1:
            logging.warning(f"File {os.path.basename(file_path)} is too short for feature extraction.")
            return None

        features = {}
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f"mfcc{i+1}_mean"], features[f"mfcc{i+1}_var"] = np.mean(mfcc[i]), np.var(mfcc[i])
        rms = librosa.feature.rms(y=y); features["rms_mean"], features["rms_var"] = np.mean(rms), np.var(rms)
        zcr = librosa.feature.zero_crossing_rate(y); features["zero_crossing_rate_mean"], features["zero_crossing_rate_var"] = np.mean(zcr), np.var(zcr)
        features["tempo"] = librosa.feature.tempo(y=y, sr=sr)[0]
        # ... Add other features as needed, this is a sample ...
        return features
    except Exception as e:
        logging.error(f"Feature extraction failed for {os.path.basename(file_path)}: {e}")
        return None

def predict_deep(features):
    order = scaler.feature_names_in_
    x = np.array([features.get(f, 0.0) for f in order]).reshape(1, -1)
    x_scaled = scaler.transform(x)
    encoded = encoder.predict(x_scaled)
    umap_embed = umap_model.transform(encoded)
    cluster = kmeans.predict(umap_embed)[0]
    return cluster_emotions.get(cluster, "Unknown")

def predict_xgb(features):
    required_features = xgb_model.get_booster().feature_names
    padded_features = {f: features.get(f, 0.0) for f in required_features}
    x = pd.DataFrame([padded_features])
    pred = xgb_model.predict(x)[0]
    return label_encoder.inverse_transform([int(pred)])[0]

def process_file_safe(file):
    """Safely processes a single file, catching exceptions."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
            temp_file.write(file.getbuffer())
            temp_path = temp_file.name

        features = extract_features(temp_path)
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()

    if features:
        deep_pred = predict_deep(features)
        xgb_pred = predict_xgb(features)
        return {"Filename": file.name, "Deep Model": deep_pred, "XGBoost Model": xgb_pred}
    else:
        return {"Filename": file.name, "Deep Model": "Processing Error", "XGBoost Model": "Processing Error"}

# --- Main App Logic with Robust State Management ---
def main():
    st.title("🎧 Dual Mood Prediction App")
    st.info("Workflow: 1. Upload all your files. 2. Click 'Start Analysis'. 3. View results.")
    
    # Initialize state variables
    if "results" not in st.session_state:
        st.session_state.results = None
    if "last_uploaded_files" not in st.session_state:
        st.session_state.last_uploaded_files = []

    uploaded_files = st.file_uploader(
        "Step 1: Upload your audio files (.mp3, .wav)",
        type=["mp3", "wav"],
        accept_multiple_files=True
    )
    
    # Detect if a new set of files has been uploaded
    if uploaded_files and uploaded_files != st.session_state.last_uploaded_files:
        st.session_state.results = None # Clear old results
        st.session_state.last_uploaded_files = uploaded_files

    if uploaded_files:
        if st.session_state.results is None:
            st.markdown(f"**{len(uploaded_files)} files staged for analysis.**")
            
            if st.button("Step 2: Start Analysis", type="primary"):
                with st.spinner("Processing files... This may take a few minutes."):
                    progress_bar = st.progress(0, text="Initializing...")
                    results_list = []
                    total_files = len(uploaded_files)
                    semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)

                    def worker(file):
                        with semaphore:
                            return process_file_safe(file)

                    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
                        future_to_file = {executor.submit(worker, file): file for file in uploaded_files}
                        completed = 0
                        for future in as_completed(future_to_file):
                            result = future.result()
                            results_list.append(result)
                            completed += 1
                            progress_bar.progress(
                                completed / total_files,
                                text=f"Processed {completed}/{total_files}: {result['Filename']}"
                            )
                    
                    st.session_state.results = pd.DataFrame(results_list)
                    progress_bar.empty()
                st.rerun() # Rerun once to display the results table cleanly
        
        else:
            st.success("✅ Analysis Complete!")
            st.dataframe(st.session_state.results)
            st.warning("To analyze a new batch of files, simply upload them above. The old results will be cleared.")

if __name__ == "__main__":
    main()