import librosa
import numpy as np
import pandas as pd
import os
import sys
from joblib import Parallel, delayed
from tqdm import tqdm

def extract_human_features(file_path):
    try:
        y, sr = librosa.load(file_path, offset=45.0, duration=30.0)  # Full-quality, mid-section
        features = {}

        # === Core Audio Features ===
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        features["rms_mean"] = np.mean(rms)
        features["rms_var"] = np.var(rms)
        features["zero_crossing_rate_mean"] = np.mean(zcr)
        features["zero_crossing_rate_var"] = np.var(zcr)
        features["tempo"] = librosa.beat.tempo(y=y, sr=sr)[0]
        features["spectral_centroid_mean"] = np.mean(centroid)
        features["spectral_centroid_var"] = np.var(centroid)
        features["spectral_bandwidth_mean"] = np.mean(bandwidth)
        features["spectral_bandwidth_var"] = np.var(bandwidth)
        features["rolloff_mean"] = np.mean(rolloff)
        features["rolloff_var"] = np.var(rolloff)
        features["perceptr_mean"] = np.mean(flatness)
        features["perceptr_var"] = np.var(flatness)
        features["chroma_stft_mean"] = np.mean(chroma)
        features["chroma_stft_var"] = np.var(chroma)

        for i in range(20):
            features[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
            features[f"mfcc{i+1}_var"] = np.var(mfcc[i])

        # === Human-Aware Features ===
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

        # === Beat Count ===
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        features["beat_count"] = len(beat_frames)

        features["filename"] = os.path.basename(file_path)
        return features

    except Exception as e:
        print(f"❌ {os.path.basename(file_path)} — {e}")
        return None

def process_folder_parallel(folder_path, n_jobs=-1):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith((".mp3", ".wav"))]
    print(f"⚙️ Processing {len(files)} files with {os.cpu_count()} cores...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_human_features)(file) for file in tqdm(files)
    )

    results = [r for r in results if r is not None]
    return pd.DataFrame(results)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗Usage: python featureExtractAdv.py <folder_path>")
        sys.exit()

    folder = sys.argv[1]
    df = process_folder_parallel(folder)

    if not df.empty:
        df.to_csv("melora_human_features.csv", index=False)
        print(f"\n✅ Saved features to melora_human_features.csv")
    else:
        print("⚠️ No features extracted.")
