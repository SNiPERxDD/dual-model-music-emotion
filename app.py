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
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment # <-- IMPORT Pydub

# Configure Streamlit
st.set_page_config(
    page_title="Dual Mood Prediction App",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Set memory limits
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "200"  # 200MB
os.environ["STREAMLIT_SERVER_MAX_MESSAGE_SIZE"] = "200"  # 200MB

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure logging
logging.basicConfig(level=logging.INFO)

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
        def _pydub_load_like_dualmodel(file_path, offset_sec=45.0, duration_sec=30.0):
            try:
                audio = AudioSegment.from_file(file_path)
                audio = audio.set_channels(1)  # mono like librosa.load default
                sr = audio.frame_rate

                start_ms = int(offset_sec * 1000)
                end_ms = start_ms + int(duration_sec * 1000)

                if start_ms >= len(audio):
                    # If file is shorter than offset, fallback to last duration window or full
                    if len(audio) > duration_sec * 1000:
                        audio = audio[-int(duration_sec * 1000):]
                    else:
                        audio = audio
                else:
                    # Clip desired segment within bounds
                    end_ms = min(end_ms, len(audio))
                    audio = audio[start_ms:end_ms]

                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                max_val = float(1 << (8 * audio.sample_width - 1))
                samples = samples / max_val  # normalize to [-1, 1]

                return samples, sr
            except Exception as e:
                logging.error(f"Audio loading error: {e}")
                raise

        y, sr = _pydub_load_like_dualmodel(file_path)
        features = {}

        # Use smaller hop lengths to reduce memory usage
        hop_length = 1024  # Increased from default 512

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
        # Pre-calculate MFCC-related features once
        for i in range(20):
            features[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
            features[f"mfcc{i+1}_var"] = np.var(mfcc[i])

        # Compute core features only once where possible
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        features["rms_mean"] = np.mean(rms)
        features["rms_var"] = np.var(rms)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
        features["zero_crossing_rate_mean"] = np.mean(zcr)
        features["zero_crossing_rate_var"] = np.var(zcr)
        
        features["tempo"] = librosa.feature.tempo(y=y, sr=sr, hop_length=hop_length)[0]

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
        features["spectral_centroid_mean"] = np.mean(centroid)
        features["spectral_centroid_var"] = np.var(centroid)

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
        features["spectral_bandwidth_mean"] = np.mean(bandwidth)
        features["spectral_bandwidth_var"] = np.var(bandwidth)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
        features["rolloff_mean"] = np.mean(rolloff)
        features["rolloff_var"] = np.var(rolloff)

        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)
        features["perceptr_mean"] = np.mean(flatness)
        features["perceptr_var"] = np.var(flatness)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        features["chroma_stft_mean"] = np.mean(chroma)
        features["chroma_stft_var"] = np.var(chroma)

        # --- Human-Aware Features ---
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr, hop_length=hop_length)
        features["tonnetz_mean"] = np.mean(tonnetz)
        features["tonnetz_var"] = np.var(tonnetz)

        harmonic, percussive = librosa.effects.hpss(y)
        features["harmonic_ratio"] = np.mean(harmonic) / (np.mean(percussive) + 1e-6)

        pitches, _ = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
        pitches = pitches[pitches > 0]
        features["pitch_std"] = np.std(pitches) if len(pitches) > 0 else 0

        mfcc_delta = librosa.feature.delta(mfcc)
        features["mfcc_delta_mean"] = np.mean(mfcc_delta)
        features["mfcc_delta_var"] = np.var(mfcc_delta)

        features["energy_score"] = features["rms_mean"] * features["spectral_centroid_mean"]

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        features["tempo_var"] = np.var(tempogram)

        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        features["beat_count"] = len(beat_frames)

        # Clean up memory
        del y, mfcc, rms, zcr, centroid, bandwidth, rolloff, flatness, chroma
        del tonnetz, harmonic, percussive, pitches, mfcc_delta, onset_env, tempogram, beat_frames
        gc.collect()

        return features
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return None

# --- [REST OF THE FILE IS UNCHANGED] ---
# ... (Prediction Functions, Process Uploaded File, Streamlit UI) ...
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
    try:
        # Check file size
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        if file.size > MAX_FILE_SIZE:
            return "File too large", "File too large"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
            temp_file.write(file.getbuffer())
            temp_path = temp_file.name
        
        features = extract_features(temp_path)
        os.remove(temp_path)
        gc.collect()  # Free up resources
        
        if features:
            return predict_deep(features), predict_xgb(features)
        return "Error", "Error"
    except Exception as e:
        logging.error(f"Error processing file {file.name}: {e}")
        return "Processing Error", "Processing Error"

# --- Streamlit UI ---
def main():
    st.title("🎧 Dual Mood Prediction App")
    st.write("Upload audio files to get emotion predictions from two different models.")
    
    # Add file upload constraints info
    st.info("📋 **Upload Guidelines:**\n- Supported formats: MP3, WAV\n- Max file size: 50MB per file\n- Max files: 10 at once")
    
    uploaded_files = st.file_uploader("Choose files", type=["mp3", "wav"], accept_multiple_files=True)

    if uploaded_files:
        # Limit number of files
        if len(uploaded_files) > 10:
            st.error("Please upload maximum 10 files at once.")
            return
            
        # Check total size
        total_size = sum(file.size for file in uploaded_files)
        if total_size > 200 * 1024 * 1024:  # 200MB total
            st.error("Total file size exceeds 200MB. Please reduce the number or size of files.")
            return
        
        results = []
        progress_bar = st.progress(0)
        
        try:
            # Reduce concurrent processing to avoid memory issues
            max_workers = min(3, len(uploaded_files))  # Limit to 3 concurrent processes
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(process_uploaded, file): file for file in uploaded_files}
                total = len(future_to_file)
                completed = 0
                for future in as_completed(future_to_file):
                    fname = future_to_file[future].name
                    try:
                        deep_pred, xgb_pred = future.result(timeout=60)  # 60 second timeout
                        results.append({
                            "Filename": fname,
                            "Deep Model": deep_pred,
                            "XGBoost Model": xgb_pred
                        })
                    except Exception as e:
                        st.warning(f"Error processing {fname}: {str(e)}")
                        results.append({
                            "Filename": fname,
                            "Deep Model": "Error",
                            "XGBoost Model": "Error"
                        })
                    completed += 1
                    progress_bar.progress(completed / total)

            st.success("✅ Done")
            st.dataframe(pd.DataFrame(results))
            
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            st.info("Try uploading fewer or smaller files.")

if __name__ == "__main__":
    main()