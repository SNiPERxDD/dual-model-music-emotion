# Music Emotion Recognition with Hybrid Modeling

**Repository:** [https://github.com/SNiPERxDD/dual-model-music-emotion.git](https://github.com/SNiPERxDD/dual-model-music-emotion.git)

**License:** All Rights Reserved

This project develops and compares two distinct machine learning pipelines for Music Emotion Recognition (MER): a supervised XGBoost classifier and an unsupervised deep learning model. The goal is to accurately classify the emotional content of music tracks, culminating in an interactive Streamlit application for real-time prediction.

## Project Overview

Music Emotion Recognition (MER) is a challenging field in music information retrieval. This project tackles the challenge by implementing a hybrid approach:

1.  **Supervised Model (XGBoost):** An `XGBoost` classifier is trained on the GTZAN dataset, which is adapted for emotion classification by mapping genres to moods. The model is fine-tuned using `GridSearchCV` and achieves an impressive **96.5% accuracy** on the test set.

2.  **Unsupervised Model (Deep Learning):** To discover inherent emotional groupings in music without relying on labels, an unsupervised pipeline was built. This model uses an **Autoencoder** for dimensionality reduction, followed by **UMAP** and **KMeans** clustering. It successfully identifies five distinct emotional clusters, which have been mapped to 'Euphoria', 'Melancholy', 'Serenity', 'Rage', and 'Dread'.

A key component of this project is the **advanced feature engineering**, which combines standard audio features with "human-aware" metrics to better capture the nuances of musical emotion.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SNiPERxDD/dual-model-music-emotion.git
    cd MLApp
    ```

2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Launch the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The application allows users to upload an audio file and see the predicted emotion from both the supervised and unsupervised models.

## Advanced Feature Engineering
This project utilizes a rich set of over 50 features extracted using `librosa`, including:
- **Standard Features:** MFCCs, Chroma, Spectral Centroid, Zero-Crossing Rate, and Tempo.
- **"Human-Aware" Features:** To better capture perceptual aspects of sound, the following were engineered:
    - **Tonnetz:** Tonal centroid features.
    - **Harmonic-Percussive Ratio:** The ratio of harmonic to percussive components.
    - **Pitch Standard Deviation:** Measures the variation in musical pitch.
    - **MFCC Delta:** The rate of change of MFCCs, capturing temporal dynamics.


## Project Structure
```
.
├── data/                 # CSV data files
│   ├── features_3_sec.csv
│   └── melora_human_features.csv (not included in repo due to copyright)
├── models/               # Saved model files
│   ├── model.pkl
│   ├── mood_xgboost_tuned.pkl
│   └── label_encoder.pkl
├── src/                  # Source code
│   ├── autodecide_modelv3.py
│   ├── evaluation_metrics.py
│   ├── featureExtractAdv.py
│   ├── labelencoder.py
│   ├── preprocess.py
│   └── TuneMoodXGBoost.py
├── app.py                # Main Streamlit application
├── .gitignore
├── README.md
├── LICENSE
└── requirements.txt
```

## Disclaimer

This project and its contents are provided **for personal, educational, and non-commercial use only**. Redistribution, commercial usage, or derivative works are **strictly prohibited** without **explicit written permission** from the author.

Some assets such as `melora_human_features.csv` are **not included** due to copyright and licensing constraints. You are responsible for ensuring compliance with all applicable laws and licenses when using or adapting this project.

All rights, including intellectual property rights in the source code, models, features, and documentation, are reserved by the author.