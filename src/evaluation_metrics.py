#!/usr/bin/env python3
"""
evaluation_metrics.py

Evaluate the XGBoost classifier and the unsupervised clustering pipeline used in the Melora project.

Requirements:
    pip install pandas scikit-learn xgboost joblib umap-learn
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.cluster import KMeans
from umap import UMAP


def evaluate_classification():
    print("=== Classification Metrics ===")

    df = pd.read_csv("features_3_sec.csv")

    # Load models
    xgb = joblib.load("mood_xgboost_tuned.pkl")
    le = joblib.load("label_encoder.pkl")

    # Map genres to moods
    genre_to_mood = {
        "blues": "Sad", "classical": "Calm", "country": "Romantic",
        "disco": "Happy", "hiphop": "Energetic", "jazz": "Calm",
        "metal": "Energetic", "pop": "Happy", "reggae": "Chill", "rock": "Energetic"
    }
    df["mood"] = df["label"].map(genre_to_mood)
    X = df.drop(columns=["filename", "length", "label", "mood"])
    y_true = le.transform(df["mood"])

    y_pred = xgb.predict(X)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

    print(f"Accuracy:             {acc:.3f}")
    print(f"Precision (weighted): {precision:.3f}")
    print(f"Recall (weighted):    {recall:.3f}")
    print(f"F1-score (weighted):  {f1:.3f}")
    print("\nConfusion Matrix (% by true class):")
    print(pd.DataFrame(cm_percent, index=le.classes_, columns=le.classes_).round(1))


def evaluate_clustering():
    print("\n=== Clustering Metrics ===")

    df = pd.read_csv("melora_human_features.csv")

    # Keep only numeric columns (drop filename etc.)
    X = df.select_dtypes(include='number')

    # Standardize features
    X_scaled = StandardScaler().fit_transform(X)

    # UMAP projection
    print("Using UMAP for dimensionality reduction.")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d = reducer.fit_transform(X_scaled)

    # KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X_2d)

    # Clustering quality metrics
    sil = silhouette_score(X_2d, labels)
    ch = calinski_harabasz_score(X_2d, labels)
    db = davies_bouldin_score(X_2d, labels)

    print(f"Silhouette Score:        {sil:.3f}")
    print(f"Calinski-Harabasz Index: {ch:.0f}")
    print(f"Davies-Bouldin Index:    {db:.3f}")


if __name__ == "__main__":
    evaluate_classification()
    evaluate_clustering()
