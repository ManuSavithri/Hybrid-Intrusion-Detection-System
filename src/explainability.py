import shap
import matplotlib.pyplot as plt

def explain_xgb(model, X_test_latent, n_explain=200):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test_latent[:n_explain])
    shap.summary_plot(shap_values, X_test_latent[:n_explain], show=True)
