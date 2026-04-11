"""
Crop Yield Prediction Model
============================
Hackathon-ready training script using Random Forest + XGBoost
Dataset: crop_yield.csv (19689 rows, Indian agriculture data)

Run: python crop_yield_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("📦 Loading dataset...")
df = pd.read_csv("crop_yield.csv")

# Strip whitespace from string columns
df["Season"] = df["Season"].str.strip()
df["Crop"] = df["Crop"].str.strip()
df["State"] = df["State"].str.strip()

print(f"   Rows: {len(df)} | Columns: {df.columns.tolist()}")


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n🔧 Encoding categorical features...")

le_crop = LabelEncoder()
le_season = LabelEncoder()
le_state = LabelEncoder()

df["Crop_enc"] = le_crop.fit_transform(df["Crop"])
df["Season_enc"] = le_season.fit_transform(df["Season"])
df["State_enc"] = le_state.fit_transform(df["State"])

# Save encoders (needed later for FastAPI inference)
joblib.dump(le_crop, "encoder_crop.pkl")
joblib.dump(le_season, "encoder_season.pkl")
joblib.dump(le_state, "encoder_state.pkl")
print("   Encoders saved: encoder_crop.pkl, encoder_season.pkl, encoder_state.pkl")


# ─────────────────────────────────────────────
# 3. PREPARE FEATURES & TARGET
# ─────────────────────────────────────────────
FEATURES = [
    "Crop_enc", "Crop_Year", "Season_enc", "State_enc",
    "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"
]
TARGET = "Yield"

X = df[FEATURES]
y = df[TARGET]

# Remove extreme outliers (top 1% yield values — they skew the model)
q99 = y.quantile(0.99)
mask = y <= q99
X, y = X[mask], y[mask]
print(f"\n🧹 After removing top 1% outliers: {len(X)} rows remaining")


# ─────────────────────────────────────────────
# 4. TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📊 Train size: {len(X_train)} | Test size: {len(X_test)}")


# ─────────────────────────────────────────────
# 5. TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
print("\n🌲 Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,       # 100 trees
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
rf_model.fit(X_train, y_train)
print("   Done!")


# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, name, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n📈 {name} Results:")
    print(f"   MAE  (Mean Absolute Error):  {mae:.4f}")
    print(f"   RMSE (Root Mean Sq Error):   {rmse:.4f}")
    print(f"   R²   (Accuracy score):       {r2:.4f}  ← closer to 1.0 is better")

evaluate(rf_model, "Random Forest", X_test, y_test)


# ─────────────────────────────────────────────
# 7. OPTIONAL: TRY XGBOOST (better accuracy)
# ─────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    print("\n⚡ Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    evaluate(xgb_model, "XGBoost", X_test, y_test)

    # Save whichever is better
    best_model = xgb_model
    joblib.dump(xgb_model, "yield_model.pkl")
    print("\n✅ Saved XGBoost model as yield_model.pkl")

except ImportError:
    print("\n⚠️  XGBoost not installed (pip install xgboost). Using Random Forest.")
    best_model = rf_model
    joblib.dump(rf_model, "yield_model.pkl")
    print("✅ Saved Random Forest model as yield_model.pkl")


# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n🔍 Feature Importance (Random Forest):")
importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
for feat, score in importances.sort_values(ascending=False).items():
    print(f"   {feat:<20} {score:.4f}")


# ─────────────────────────────────────────────
# 9. SAMPLE PREDICTION (for testing)
# ─────────────────────────────────────────────
print("\n🧪 Sample Prediction Test:")
# Rice, 2015, Kharif, Uttar Pradesh
sample = {
    "Crop": "Rice",
    "Crop_Year": 2015,
    "Season": "Kharif",
    "State": "Uttar Pradesh",
    "Area": 5000,
    "Annual_Rainfall": 1000,
    "Fertilizer": 500000,
    "Pesticide": 15000,
}

# Encode the sample
try:
    sample_enc = [[
        le_crop.transform([sample["Crop"]])[0],
        sample["Crop_Year"],
        le_season.transform([sample["Season"]])[0],
        le_state.transform([sample["State"]])[0],
        sample["Area"],
        sample["Annual_Rainfall"],
        sample["Fertilizer"],
        sample["Pesticide"],
    ]]
    pred = best_model.predict(sample_enc)[0]
    print(f"   Input: {sample}")
    print(f"   Predicted Yield: {pred:.4f} tons/hectare")
except Exception as e:
    print(f"   Sample prediction failed: {e}")

print("\n✅ Training complete! Files saved:")
print("   - yield_model.pkl       ← main model for FastAPI")
print("   - encoder_crop.pkl")
print("   - encoder_season.pkl")
print("   - encoder_state.pkl")