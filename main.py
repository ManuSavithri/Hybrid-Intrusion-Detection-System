import numpy as np
import pandas as pd
import joblib
from src.preprocessing import load_and_preprocess
from src.autoencoder import build_autoencoder, train_autoencoder
from src.threshold import compute_mse_on, robust_threshold
from src.classifiers import train_xgb, train_nn
from src.pipeline import combined_pipeline
from src.explainability import explain_xgb
import matplotlib.pyplot as plt

# ================== CONFIG ==================
DATA_PATH = "cybersecurity_attacks.csv"
RANDOM_STATE = 42

# ================== 1. PREPROCESS ==================
X_train, X_test, y_train_bin, y_test_bin, y_train_type, y_test_type, scaler, X_raw, y_attack_type = load_and_preprocess(DATA_PATH, RANDOM_STATE)

# Save feature names
feature_names = list(X_raw.columns)
joblib.dump(feature_names, "feature_names.pkl")
print("Saved feature names.")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("Saved scaler.")

# ================== 2. AUTOENCODER ==================
autoencoder, encoder = build_autoencoder(X_train.shape[1])
X_train_normal = X_train[y_train_bin == 0]
train_autoencoder(autoencoder, X_train_normal)

# Save encoder (autoencoder bottleneck)
encoder.save("encoder_model.h5")
print("Saved encoder model.")

# ================== 3. THRESHOLDING ==================
median_threshold, median_f1, _, _ = robust_threshold(autoencoder, X_train, y_train_bin)
print("Robust threshold:", median_threshold, "Median F1:", median_f1)

# Test anomaly detection
mse_test = compute_mse_on(autoencoder, X_test)
anomaly_preds = (mse_test > median_threshold).astype(int)

# ================== 4. LATENT FEATURES ==================
X_train_latent = encoder.predict(X_train)
X_test_latent = encoder.predict(X_test)

# ================== 5. CLASSIFIERS ==================
# XGBoost on latent features
xgb_latent, _ = train_xgb(X_train_latent, y_train_type, X_test_latent, y_test_type, "(latent)")

# Optional: XGBoost on raw features
xgb_raw, _ = train_xgb(X_train, y_train_type, X_test, y_test_type, "(raw)")

# Dense NN classifier
nn_model, _ = train_nn(X_train_latent, y_train_type, X_test_latent, y_test_type, len(np.unique(y_train_type)))

# Save XGBoost latent model for Streamlit
joblib.dump(xgb_latent, "xgb_model.pkl")
print("Saved XGBoost model.")

# ================== 6. COMBINED PIPELINE ==================
combined_pipeline(anomaly_preds, X_test_latent, xgb_latent, y_test_bin, y_test_type)

# ================== 7. EXPLAINABILITY ==================
explain_xgb(xgb_latent, X_test_latent)
