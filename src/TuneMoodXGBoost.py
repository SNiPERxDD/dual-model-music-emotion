# TuneMoodXGBoost.py
# Hyperparameter tuning of XGBoost mood model using GridSearchCV

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the 3-sec feature dataset
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

# Prepare features and labels
X = df.drop(columns=["filename", "length", "label", "mood"])
y = LabelEncoder().fit_transform(df["mood"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define parameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

# Initialize and run grid search
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring="f1_weighted", verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# Best model evaluation
y_pred = grid.best_estimator_.predict(X_test)
print("\n✅ Best Params:", grid.best_params_)
print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(grid.best_estimator_, "mood_xgboost_tuned.pkl")
print("\n✅ Saved tuned model as mood_xgboost_tuned.pkl")
