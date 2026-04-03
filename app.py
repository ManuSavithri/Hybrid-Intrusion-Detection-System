# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow.keras.models import load_model

# -------------------- Page config --------------------
st.set_page_config(
    page_title="🚨 Explainable Anomaly Detection in Network Traffic",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Helper functions --------------------
def get_encoder_effective_weights(encoder_model):
    """
    Return an effective input->latent weight matrix W_eff of shape
    (input_dim, latent_dim) by multiplying Dense kernels of encoder layers.
    """
    kernels = []
    for layer in encoder_model.layers:
        try:
            w = layer.get_weights()
            if len(w) >= 1:
                kernel = w[0]
                kernels.append(kernel)
        except Exception:
            continue
    if len(kernels) == 0:
        raise ValueError("No kernels found in encoder layers.")

    W_eff = kernels[0]
    for k in kernels[1:]:
        W_eff = np.dot(W_eff, k)
    return W_eff

def map_latent_shap_to_input(shap_vals_latent, encoder_model, input_feature_names):
    """
    Map SHAP values on latent dims back to input features.
    """
    W_eff = get_encoder_effective_weights(encoder_model)

    arr = np.array(shap_vals_latent)
    single = False
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        single = True

    mapped = np.dot(arr, W_eff.T)

    if single:
        return mapped.flatten(), input_feature_names
    else:
        return mapped, input_feature_names

# -------------------- Load artifacts --------------------
@st.cache_resource
def load_artifacts():
    encoder = load_model("encoder_model.h5")          # Autoencoder encoder
    xgb_model = joblib.load("xgb_model.pkl")          # XGBoost classifier
    scaler = joblib.load("scaler.pkl")                # Scaler
    feature_names = joblib.load("feature_names.pkl")  # Feature order
    return encoder, xgb_model, scaler, feature_names

try:
    encoder, xgb_model, scaler, feature_names = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model artifacts: {e}")
    st.stop()

# -------------------- Header --------------------
st.title("🚨 Explainable Anomaly Detection in Network Traffic")
st.markdown("""
This dashboard uses **Autoencoder + XGBoost** to detect anomalies in network traffic.  
It focuses on **explainability (XAI)** so security teams understand *why* a traffic flow is flagged as suspicious.
""")

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("📂 Upload a CSV of network traffic", type="csv")
if not uploaded_file:
    st.info("Upload a CSV file (same schema as training). Example: `synthetic_data.csv`.")
    st.stop()

df = pd.read_csv(uploaded_file)

# -------------------- Feature alignment --------------------
X_raw = df.copy()
missing = [c for c in feature_names if c not in X_raw.columns]
extra = [c for c in X_raw.columns if c not in feature_names]

if missing:
    st.warning(f"⚠️ Missing {len(missing)} training columns — filling zeros. Example: {missing[:5]}")
    for c in missing:
        X_raw[c] = 0

X_raw = X_raw[feature_names]

if extra:
    st.info(f"ℹ️ Ignoring {len(extra)} extra uploaded columns (first few): {extra[:5]}")

# -------------------- Scale & Encode --------------------
try:
    X_scaled = scaler.transform(X_raw)
    X_latent = encoder.predict(X_scaled)
except Exception as e:
    st.error(f"❌ Error processing features: {e}")
    st.stop()

latent_names = [f"z_{i}" for i in range(X_latent.shape[1])]
X_latent_df = pd.DataFrame(X_latent, columns=latent_names)

# -------------------- Predictions & Risk --------------------
try:
    preds = xgb_model.predict(X_latent)
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.stop()

df["Predicted_Attack"] = preds

if hasattr(xgb_model, "predict_proba"):
    proba = xgb_model.predict_proba(X_latent)
    df["Risk_Score"] = (proba.max(axis=1) * 100).astype(int)
else:
    df["Risk_Score"] = [100 if p != "Normal" else 0 for p in preds]

# -------------------- Sidebar Risk Analysis --------------------
st.sidebar.header("📊 Risk Analysis")
high_risk = df[df["Risk_Score"] >= 70].shape[0]
moderate_risk = df[(df["Risk_Score"] >= 40) & (df["Risk_Score"] < 70)].shape[0]
low_risk = df[df["Risk_Score"] < 40].shape[0]

st.sidebar.metric("🔥 High Risk", high_risk)
st.sidebar.metric("⚠️ Moderate Risk", moderate_risk)
st.sidebar.metric("✅ Low Risk", low_risk)

risk_counts = {"High": high_risk, "Moderate": moderate_risk, "Low": low_risk}
fig_risk = px.pie(
    names=list(risk_counts.keys()),
    values=list(risk_counts.values()),
    color=list(risk_counts.keys()),
    color_discrete_map={"High": "red", "Moderate": "orange", "Low": "green"},
    title="Risk Distribution"
)
st.sidebar.plotly_chart(fig_risk, use_container_width=True)

# -------------------- Tabs --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Preview",
    "🔮 Predictions & Risk Scores",
    "📊 Predicted Class Summary",
    "🧠 Explainability (XAI)",
    "🔍 Debugging Predictions"
])

# -------------------- Tab 1 --------------------
with tab1:
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.write(f"Rows: {len(df)}, Expected features: {len(feature_names)}")

# -------------------- Tab 2 --------------------
with tab2:
    st.subheader("Predictions & Risk Scores")
    styled = df.head(50).style.background_gradient(subset=["Risk_Score"], cmap="Reds")
    st.dataframe(styled, use_container_width=True)

    pred_counts = df["Predicted_Attack"].value_counts()
    fig_pred = px.bar(
        x=pred_counts.index, y=pred_counts.values, color=pred_counts.values,
        color_continuous_scale="Viridis",
        labels={"x": "Attack Type", "y": "Count"},
        title="Prediction Distribution"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.download_button(
        "⬇️ Download Predictions CSV",
        df.to_csv(index=False).encode("utf-8"),
        "predictions.csv"
    )

# -------------------- Tab 3 --------------------
with tab3:
    st.subheader("Predicted Attack Summary")
    class_counts = df["Predicted_Attack"].value_counts()
    class_perc = (class_counts / len(df) * 100).round(2)
    summary_df = pd.DataFrame({"Count": class_counts, "Percentage (%)": class_perc})
    st.dataframe(summary_df.style.set_properties(**{"background-color": "#f0f0f0"}))

    fig = px.bar(
        summary_df, x=summary_df.index, y="Count",
        color="Percentage (%)", color_continuous_scale="OrRd",
        title="Predicted Attack Types Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Tab 4 (Explainability) --------------------
with tab4:
    st.subheader("Explainability with SHAP (Global & Local)")

    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals_global = explainer.shap_values(X_latent_df)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_vals_global, X_latent_df, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ Could not compute SHAP global values: {e}")

    st.markdown("---")
    st.write("🔍 Explain a Single Prediction")
    max_idx = max(0, X_latent_df.shape[0] - 1)
    row_idx = st.number_input("Row index to explain", min_value=0, max_value=max_idx, value=0, step=1)

    try:
        shap_row_vals = explainer.shap_values(X_latent_df.iloc[row_idx:row_idx+1])
        row_df = X_latent_df.iloc[row_idx:row_idx+1]

        if isinstance(shap_row_vals, list):
            try:
                pred_class = int(preds[row_idx])
            except Exception:
                pred_class = 0
            shap_for_class = np.array(shap_row_vals[pred_class]).reshape(-1)
        else:
            shap_for_class = np.array(shap_row_vals).reshape(-1)

        mapped_contrib, input_feat_names = map_latent_shap_to_input(shap_for_class, encoder, feature_names)
        abs_contrib = pd.Series(mapped_contrib, index=input_feat_names).abs().sort_values(ascending=False)

        top_n = min(10, len(abs_contrib))
        top_feats = abs_contrib.iloc[:top_n].index.tolist()
        signed_contribs = pd.Series(mapped_contrib, index=input_feat_names).loc[top_feats]

        fig2 = px.bar(
            x=signed_contribs.index,
            y=signed_contribs.values,
            color=signed_contribs.values,
            labels={"x": "Feature", "y": "Mapped Contribution"},
            title=f"Top {top_n} Original Feature Contributions for Row {row_idx} (approx)"
        )
        st.plotly_chart(fig2, use_container_width=True)

        contrib_df = pd.DataFrame({
            "feature": signed_contribs.index,
            "signed_contribution": signed_contribs.values,
            "abs_contribution": np.abs(signed_contribs.values)
        }).reset_index(drop=True)
        st.table(contrib_df)

    except Exception as e:
        st.warning(f"⚠️ Could not compute mapped SHAP explanation: {e}")

# -------------------- Tab 5 --------------------
with tab5:
    st.subheader("🔍 Debugging Predictions vs Ground Truth")

    if "Attack Type" in df.columns:
        y_true = df["Attack Type"].astype(str).values
        y_pred_local = preds.astype(str) if isinstance(preds[0], (int, np.integer, str)) else [str(p) for p in preds]

        comparison_df = pd.DataFrame({
            "Actual_Label": y_true[:len(y_pred_local)],
            "Predicted_Label": y_pred_local
        })

        st.dataframe(comparison_df.head(50), use_container_width=True)

        st.write("### Prediction Distribution")
        pred_counts = pd.Series(y_pred_local).value_counts()
        fig_debug = px.bar(
            x=pred_counts.index, 
            y=pred_counts.values, 
            text=pred_counts.values,
            title="Predicted Classes Count",
            labels={"x": "Class", "y": "Count"}
        )
        st.plotly_chart(fig_debug, use_container_width=True)

        st.download_button(
            "⬇️ Download Actual vs Predicted",
            comparison_df.to_csv(index=False).encode("utf-8"),
            "comparison.csv"
        )
    else:
        st.info("⚠️ No ground-truth column found in uploaded data.")
