import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import os
import random
import warnings

# === Seed Everything ===
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# === MacOS GPU Fix ===
tf.config.set_visible_devices([], 'GPU')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# === Load Data ===
df = pd.read_csv("melora_human_features.csv")
filenames = df["filename"]
X = df.drop(columns=["filename"], errors="ignore")

# === Scale ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Autoencoder Model ===
inp = Input(shape=(X_scaled.shape[1],))
x = Dense(512)(inp)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(256)(x)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128)(x)
x = LeakyReLU()(x)

encoded = Dense(64)(x)
encoded = LeakyReLU()(encoded)

x = Dense(128)(encoded)
x = LeakyReLU()(x)

x = Dense(256)(x)
x = LeakyReLU()(x)

x = Dense(512)(x)
x = LeakyReLU()(x)

out = Dense(X_scaled.shape[1], activation='linear')(x)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=1e-5), loss='mse')

early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
autoencoder.fit(X_scaled, X_scaled, epochs=300, batch_size=64, shuffle=True, verbose=1, callbacks=[early_stop])

# === Encoder Output ===
encoder = Model(inp, encoded)
encoded_features = encoder.predict(X_scaled)

# === UMAP Grid Search ===
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.*")
best_score = -1
for n in [5, 6, 8, 10]:
    for d in [0.0, 0.001, 0.005, 0.01]:
        reducer = umap.UMAP(n_neighbors=n, min_dist=d, n_components=2, random_state=42)
        umap_embed = reducer.fit_transform(encoded_features)
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(umap_embed)

        sil = silhouette_score(umap_embed, clusters)
        ch = calinski_harabasz_score(umap_embed, clusters)
        db = davies_bouldin_score(umap_embed, clusters)

        print(f"n_neighbors={n}, min_dist={d} → Silhouette: {sil:.3f} | CH: {ch:.2f} | DB: {db:.2f}")

        if sil > best_score:
            best_score = sil
            best_config = (n, d)
            best_umap = reducer  # Save the UMAP model (not the embedding)
            best_clusters = clusters
            best_kmeans = kmeans

print(f"\n✅ Best UMAP Config: n_neighbors={best_config[0]}, min_dist={best_config[1]} → Silhouette: {best_score:.3f}")

# === Save Results ===
# Retrieve the embedding from the UMAP model
embedding = best_umap.embedding_
df["Cluster"] = best_clusters
df["UMAP1"] = embedding[:, 0]
df["UMAP2"] = embedding[:, 1]
df["filename"] = filenames

# Compute distance to centroid in the UMAP space
centroids = best_kmeans.cluster_centers_
df["DistToCentroid"] = [
    np.linalg.norm(embedding[i] - centroids[c])
    for i, c in enumerate(best_clusters)
]

joblib.dump(df, "melora_clustered_final.pkl")

# === Plot with Centroids ===
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df["UMAP1"], df["UMAP2"], c=df["Cluster"], cmap="tab10", s=30, label="Songs")

# Plot centroids
for idx, (cx, cy) in enumerate(centroids):
    plt.scatter(cx, cy, marker='X', s=200, edgecolors='black', facecolors='red', linewidths=2)
    plt.text(cx, cy, f'C{idx}', fontsize=10, fontweight='bold', ha='center', va='center')

plt.title("Melora Clusters (Optimized UMAP + Autoencoder)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("melora_optimized_clusters_with_centroids.png")
plt.show()

# === Print Central Songs ===
for c in sorted(df["Cluster"].unique()):
    print(f"\n🎧 Cluster {c} (Most Central):")
    top = df[df["Cluster"] == c].sort_values("DistToCentroid").head(10)["filename"]
    for song in top:
        print(f"  - {song}")

# === Auto-label emotions ===
cluster_emotions = {
    0: "Euphoria",
    1: "Melancholy",
    2: "Serenity",
    3: "Rage",
    4: "Dread"
}
df["Emotion"] = df["Cluster"].map(cluster_emotions)

# === Sort by cluster and closeness to centroid ===
df_sorted = df.sort_values(by=["Cluster", "DistToCentroid"])

# === Save everything to CSV ===
df_sorted.to_csv("melora_emotion_clusters1.csv", index=False)
print("\n✅ Saved full feature CSV: melora_emotion_clusters.csv (cluster-wise, closest first)")

# === Save the Pipeline ===
# This dictionary contains all components needed for later predictions.
model_pipeline = {
    "scaler": scaler,
    "encoder": encoder,
    "umap": best_umap,
    "kmeans": best_kmeans,
    "cluster_emotions": cluster_emotions
}

joblib.dump(model_pipeline, "model.pkl")
print("✅ Pipeline saved to model.pkl")
