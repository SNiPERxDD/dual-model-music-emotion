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
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment

# --- Configuration ---
# Set the maximum number of files to process at the same time.
# A value between 2 and 4 is a safe starting point for Streamlit Cloud.
MAX_CONCURRENT_JOBS = 2

# --- Load Pre-trained Models ---
@st.cache_resource
def load_models():
    """Loads all the models and encoders into memory once."""
    deep_model = joblib.load("models/model.pkl")
    xgb_model = joblib.load("models/mood_xgboost_tuned.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return deep_model, xgb_model, label_encoder

try:
    deep_model, xgb_model, label_encoder = load_models()
    scaler = deep_model["scaler"]
    encoder = deep_model["encoder"]
    umap_model = deep_model["umap"]
    kmeans = deep_model["kmeans"]
    cluster_emotions = deep_model["cluster_emotions"]
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'model.pkl', 'mood_xgboost_tuned.pkl', and 'label_encoder.pkl' are in the 'models' directory.")
    st.stop()


# --- Feature Extraction & Prediction (No changes needed here) ---
def extract_features(file_path):
    try:
        def _pydub_load_like_dualmodel(file_path, offset_sec=45.0, duration_sec=30.0):
                audio = AudioSegment.from_file(file_path)
                audio = audio.set_channels(1)
                sr = audio.frame_rate
                start_ms, end_ms = int(offset_sec * 1000), int((offset_sec + duration_sec) * 1000)
                if start_ms >= len(audio):
                    audio = audio[-int(duration_sec * 1000):] if len(audio) > duration_sec * 1000 else audio
                else:
                    audio = audio[start_ms:min(end_ms, len(audio))]
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / float(1 << (8 * audio.sample_width - 1))
                return samples, sr

        y, sr = _pydub_load_like_dualmodel(file_path)
        features = {}
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f"mfcc{i+1}_mean"], features[f"mfcc{i+1}_var"] = np.mean(mfcc[i]), np.var(mfcc[i])
        rms = librosa.feature.rms(y=y); features["rms_mean"], features["rms_var"] = np.mean(rms), np.var(rms)
        zcr = librosa.feature.zero_crossing_rate(y); features["zero_crossing_rate_mean"], features["zero_crossing_rate_var"] = np.mean(zcr), np.var(zcr)
        features["tempo"] = librosa.feature.tempo(y=y, sr=sr)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr); features["spectral_centroid_mean"], features["spectral_centroid_var"] = np.mean(centroid), np.var(centroid)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr); features["spectral_bandwidth_mean"], features["spectral_bandwidth_var"] = np.mean(bandwidth), np.var(bandwidth)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr); features["rolloff_mean"], features["rolloff_var"] = np.mean(rolloff), np.var(rolloff)
        return features
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
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

# --- Streamlit UI ---
def main():
    st.title("🎧 Dual Mood Prediction App")
    st.write("Upload audio files to get emotion predictions. Processing is rate-limited for stability.")

    # --- ROBUST STATE INITIALIZATION ---
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    if 'results' not in st.session_state:
        st.session_state.results = pd.DataFrame()
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    # --- FILE UPLOADER WITH DYNAMIC KEY ---
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["mp3", "wav"],
        accept_multiple_files=True,
        key=str(st.session_state.uploader_key) # The key is crucial for resetting
    )

    # --- PROCESSING LOGIC ---
    # This block runs only when there are files and they haven't been processed yet.
    if uploaded_files and not st.session_state.processing_done:
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
        st.session_state.processing_done = True
        progress_bar.progress(1.0, text="✅ Processing Complete!")

    # --- DISPLAY AND RESET LOGIC ---
    # This block runs only after processing is finished.
    if st.session_state.processing_done:
        st.success("Analysis complete. See results below.")
        st.dataframe(st.session_state.results)

        # The "Clear" button now correctly resets the state and the uploader
        if st.button("Analyze New Files"):
            st.session_state.processing_done = False
            st.session_state.results = pd.DataFrame()
            st.session_state.uploader_key += 1 # Changing the key forces a widget reset
            st.rerun() # A safe rerun here is fine, as it's a controlled state change

if __name__ == "__main__":
    main()