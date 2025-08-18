# recreate_label_encoder.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("features_3_sec.csv")

# Genre to mood mapping
genre_to_mood = {
    "blues": "Sad",
    "classical": "Calm",
    "country": "Romantic",
    "disco": "Happy",
    "hiphop": "Energetic",
    "jazz": "Calm",
    "metal": "Energetic",
    "pop": "Happy",
    "reggae": "Chill",
    "rock": "Energetic"
}

df["mood"] = df["label"].map(genre_to_mood)
le = LabelEncoder()
le.fit(df["mood"])
joblib.dump(le, "label_encoder.pkl")
print("✅ Saved label_encoder.pkl")
