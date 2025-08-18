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
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- Load Pre-trained Models ---
@st.cache_resource
def load_models():
    deep_model = joblib.load("models/model.pkl")
    xgb_model = joblib.load("models/mood_xgboost_tuned.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return deep_model, xgb_model, label_encoder

deep_model, xgb_model, label_encoder = load_models()
scaler = deep_model["scaler"]
encoder = deep_model["encoder"]
umap_model = deep_model["umap"]
kmeans = deep_model["kmeans"]
cluster_emotions = deep_model["cluster_emotions"]

# --- Feature Extraction ---
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, offset=45.0, duration=30.0, sr=None)
        features = {}

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        # Pre-calculate MFCC-related features once
        for i in range(20):
            features[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
            features[f"mfcc{i+1}_var"] = np.var(mfcc[i])

        # Compute core features only once where possible
        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = np.mean(rms)
        features["rms_var"] = np.var(rms)
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zero_crossing_rate_mean"] = np.mean(zcr)
        features["zero_crossing_rate_var"] = np.var(zcr)
        features["tempo"] = librosa.beat.tempo(y=y, sr=sr)[0]

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid_mean"] = np.mean(centroid)
        features["spectral_centroid_var"] = np.var(centroid)

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = np.mean(bandwidth)
        features["spectral_bandwidth_var"] = np.var(bandwidth)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features["rolloff_mean"] = np.mean(rolloff)
        features["rolloff_var"] = np.var(rolloff)

        flatness = librosa.feature.spectral_flatness(y=y)
        features["perceptr_mean"] = np.mean(flatness)
        features["perceptr_var"] = np.var(flatness)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features["chroma_stft_mean"] = np.mean(chroma)
        features["chroma_stft_var"] = np.var(chroma)

        # --- Human-Aware Features ---
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features["tonnetz_mean"] = np.mean(tonnetz)
        features["tonnetz_var"] = np.var(tonnetz)

        harmonic, percussive = librosa.effects.hpss(y)
        features["harmonic_ratio"] = np.mean(harmonic) / (np.mean(percussive) + 1e-6)

        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]
        features["pitch_std"] = np.std(pitches) if len(pitches) > 0 else 0

        mfcc_delta = librosa.feature.delta(mfcc)
        features["mfcc_delta_mean"] = np.mean(mfcc_delta)
        features["mfcc_delta_var"] = np.var(mfcc_delta)

        features["energy_score"] = features["rms_mean"] * features["spectral_centroid_mean"]

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        features["tempo_var"] = np.var(tempogram)

        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        features["beat_count"] = len(beat_frames)

        return features
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return None

# --- Prediction Functions ---
def predict_deep(features):
    order = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else sorted(features.keys())
    x = np.array([features[f] for f in order if f in features]).reshape(1, -1)
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
    mood_label = label_encoder.inverse_transform([int(pred)])[0]
    return mood_label

# --- Process Uploaded File ---
def process_uploaded(file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.getbuffer())
        temp_path = temp_file.name
    features = extract_features(temp_path)
    os.remove(temp_path)
    gc.collect()  # Free up resources
    if features:
        return predict_deep(features), predict_xgb(features)
    return "Error", "Error"

# --- Streamlit UI ---
def main():
    st.title("🎧 Dual Mood Prediction App")
    st.write("Upload audio files to get emotion predictions from two different models.")
    uploaded_files = st.file_uploader("Choose files", type=["mp3", "wav"], accept_multiple_files=True)

    if uploaded_files:
        results = []
        progress_bar = st.progress(0)

        # Use ThreadPoolExecutor to parallelize file processing
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(process_uploaded, file): file for file in uploaded_files}
            total = len(future_to_file)
            completed = 0
            for future in as_completed(future_to_file):
                fname = future_to_file[future].name
                deep_pred, xgb_pred = future.result()
                results.append({
                    "Filename": fname,
                    "Deep Model": deep_pred,
                    "XGBoost Model": xgb_pred
                })
                completed += 1
                progress_bar.progress(completed / total)

        st.success("✅ Done")
        st.dataframe(pd.DataFrame(results))

if __name__ == "__main__":
    main()
