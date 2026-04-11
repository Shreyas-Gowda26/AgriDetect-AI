"""
Crop Recommendation Model
==========================
Classification model — predicts which crop to grow based on soil & weather.
Run: python crop_recommendation_model.py

Output: ml_models/crop_rec_model.pkl + ml_models/crop_rec_scaler.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("📦 Loading dataset...")
df = pd.read_csv("Crop_recommendation.csv")
print(f"   Rows: {len(df)} | Crops: {df['label'].nunique()}")


# ─────────────────────────────────────────────
# 2. FEATURES & TARGET
# ─────────────────────────────────────────────
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET = 'label'

X = df[FEATURES]
y = df[TARGET]


# ─────────────────────────────────────────────
# 3. ENCODE TARGET LABELS
# ─────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n🏷️  Crops encoded: {list(le.classes_)}")

# Save label encoder
import os
os.makedirs("ml_models", exist_ok=True)
joblib.dump(le, "ml_models/crop_rec_label_encoder.pkl")
print("   Saved: ml_models/crop_rec_label_encoder.pkl")


# ─────────────────────────────────────────────
# 4. SCALE FEATURES (important for this dataset)
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "ml_models/crop_rec_scaler.pkl")
print("   Saved: ml_models/crop_rec_scaler.pkl")


# ─────────────────────────────────────────────
# 5. TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")


# ─────────────────────────────────────────────
# 6. TRAIN RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────
print("\n🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"   Accuracy: {rf_acc * 100:.2f}%")


# ─────────────────────────────────────────────
# 7. OPTIONAL: TRY XGBOOST
# ─────────────────────────────────────────────
best_model = rf_model
best_acc = rf_acc

try:
    from xgboost import XGBClassifier
    print("\n⚡ Training XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    print(f"   Accuracy: {xgb_acc * 100:.2f}%")

    if xgb_acc >= rf_acc:
        best_model = xgb_model
        best_acc = xgb_acc
        print("   → XGBoost wins!")
    else:
        print("   → Random Forest wins!")

except ImportError:
    print("⚠️  XGBoost not installed, using Random Forest.")


# ─────────────────────────────────────────────
# 8. SAVE BEST MODEL
# ─────────────────────────────────────────────
joblib.dump(best_model, "ml_models/crop_rec_model.pkl")
print(f"\n✅ Best model saved: ml_models/crop_rec_model.pkl (Accuracy: {best_acc * 100:.2f}%)")


# ─────────────────────────────────────────────
# 9. DETAILED REPORT
# ─────────────────────────────────────────────
print("\n📈 Classification Report (Random Forest):")
print(classification_report(y_test, rf_preds, target_names=le.classes_))


# ─────────────────────────────────────────────
# 10. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("🔍 Feature Importance:")
importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
for feat, score in importances.sort_values(ascending=False).items():
    print(f"   {feat:<15} {score:.4f}")


# ─────────────────────────────────────────────
# 11. SAMPLE PREDICTION
# ─────────────────────────────────────────────
print("\n🧪 Sample Prediction:")
sample = {
    "N": 90, "P": 42, "K": 43,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
}
sample_scaled = scaler.transform([list(sample.values())])
pred_encoded = best_model.predict(sample_scaled)[0]
pred_crop = le.inverse_transform([pred_encoded])[0]
print(f"   Input: {sample}")
print(f"   Recommended Crop: {pred_crop}")

print("\n✅ Files saved in ml_models/:")
print("   - crop_rec_model.pkl")
print("   - crop_rec_scaler.pkl")
print("   - crop_rec_label_encoder.pkl")