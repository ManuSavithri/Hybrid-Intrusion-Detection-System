# 🛡️ Hybrid Intrusion Detection System with Explainable AI

An intelligent network intrusion detection system that combines **Deep Learning, Anomaly Detection, XGBoost, and Explainable AI (XAI)** to identify suspicious network activity and classify potential cyberattacks.

The project uses an Autoencoder to learn compact representations of network traffic, machine-learning classifiers to identify attack categories, and SHAP to help interpret model predictions. An interactive Streamlit dashboard allows users to upload network traffic data, examine predictions, and explore risk assessments.

## 📌 Project Overview

Traditional intrusion detection systems can struggle with complex traffic patterns, evolving attack techniques, and the need to explain why an event was flagged.

This project explores a hybrid approach that combines neural-network-based feature learning with supervised classification and model explainability.

The system is designed to:
- Detect anomalous patterns in network traffic using reconstruction error.
- Learn compressed representations of traffic features using an Autoencoder.
- Classify network traffic into attack categories using XGBoost.
- Interpret classifier behavior using SHAP-based feature importance.
- Present predictions and risk assessments through an interactive dashboard.

**Project focus:** Network intrusion detection, anomaly detection, multiclass attack classification, and explainable machine learning.

## 🎯 Objectives

- Develop a machine-learning pipeline for identifying potentially malicious network traffic.
- Explore the combination of unsupervised representation learning and supervised classification.
- Use reconstruction error and threshold-based anomaly detection to distinguish normal traffic from suspicious activity.
- Apply explainability techniques to improve the interpretability of model predictions.
- Build a practical interface for analyzing uploaded network traffic datasets.

## 🧠 System Architecture

The project consists of a model-training pipeline and a Streamlit-based inference dashboard.

### Training Pipeline

```text
Network Traffic Dataset
          |
          v
Data Preprocessing
          |
          v
Categorical Encoding
          |
          v
Feature Scaling
          |
          +--------------------------+
          |                          |
          v                          v
     Autoencoder              Classification
          |                          |
          v                          |
 Learn Normal Patterns               |
          |                          |
          v                          |
 Reconstruction Error               |
          |                          |
          v                          |
 Anomaly Threshold                   |
          |                          |
          v                          |
     Anomaly Flags                   |
                                     |
              Encoder                |
                 |                   |
                 v                   |
         Latent Representations      |
                 |                   |
                 v                   |
          XGBoost Classifier <-------+
                 |
                 v
       Model Evaluation
                 |
                 v
        SHAP Explainability
```

### Dashboard 


<img width="513" height="234" alt="Dashboard" src="https://github.com/user-attachments/assets/16ace177-2c4a-4a2e-bb94-63cb7812de18" />
<img width="409" height="254" alt="Dashboard Col1" src="https://github.com/user-attachments/assets/84d6aae7-0743-4250-98ad-fadfb9277490" />
<img width="409" height="254" alt="Dashboard Col2" src="https://github.com/user-attachments/assets/79a99318-abf6-4542-8933-6baa5263fdec" />

---

## ⚙️ How It Works

### 1. Data Preprocessing

The input dataset is loaded and prepared for model training.

The preprocessing module:
- Separates the target labels from input features.
- Converts the attack label into a binary target: normal traffic or attack traffic.
- Encodes categorical input features into numerical representations.
- Encodes attack categories for multiclass classification.
- Splits the data into training and testing sets.
- Applies MinMaxScaler to normalize feature values.

Timestamp and source/destination IP address fields are excluded from the training features in the current implementation.

### 2. Autoencoder-Based Anomaly Detection

An Autoencoder is trained to reconstruct normal network traffic patterns.

The neural network consists of:
- An encoder with a 64-unit hidden layer.
- A 32-dimensional bottleneck representation.
- A decoder that reconstructs the input features.

The model minimizes Mean Squared Error (MSE) between the original input and its reconstruction.

The reconstruction error is calculated as:

`MSE = Mean((Original Features - Reconstructed Features)²)`

A threshold is then used to identify observations with higher reconstruction errors as potential anomalies.

The implementation includes repeated threshold estimation and uses the median selected threshold to make the process less dependent on a single threshold-selection run.

### 3. Latent Feature Extraction

The trained encoder transforms the original network traffic features into a compact, 32-dimensional representation.

These latent features are used as inputs to the downstream classifiers.

This approach explores whether learned representations can capture useful patterns that improve attack classification.

### 4. XGBoost Attack Classification

The project uses XGBoost to classify network traffic into encoded attack categories.

Two configurations are evaluated during training:

- **XGBoost on latent features:** Uses the Autoencoder's compressed representations.
- **XGBoost on scaled input features:** Uses the original preprocessed features.

A separate dense neural-network classifier is also trained on the latent representations for comparison.

Classification reports are generated to evaluate model performance.

### 5. Combined Detection Pipeline

The training pipeline also evaluates a combined detection approach.

Its logic is:

- If the anomaly detector identifies an observation as normal, the pipeline assigns the normal class.
- If the observation is flagged as anomalous, the pipeline uses XGBoost to predict its attack category.

The resulting binary detection and multiclass classification reports help evaluate this combined approach.

### 6. Explainable AI with SHAP

The project uses SHAP (SHapley Additive exPlanations) to investigate which latent features contribute to XGBoost predictions.

The explainability module generates a SHAP summary plot to visualize feature contributions across the analyzed samples.

This provides insight into classifier behavior instead of treating its predictions as completely opaque.

The current SHAP analysis operates on latent features. Consequently, its feature labels correspond to encoded dimensions rather than directly identifying the original network traffic fields.

### 7. Interactive Streamlit Dashboard

The dashboard provides an interface for analyzing network traffic through a CSV upload.

Its capabilities include:

- **Data Preview:** Inspect uploaded records.
- **Predictions and Risk Scores:** View predicted attack categories and calculated risk scores.
- **Predicted Class Summary:** Explore the distribution of predictions.
- **Explainability:** Examine model interpretation visualizations.
- **Debugging Predictions:** Investigate prediction-related information.
- **Risk Analysis:** View high-, moderate-, and low-risk record counts.

The dashboard aligns uploaded data with the saved training feature order, fills missing training columns with zeros, and ignores extra columns.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Autoencoder and neural-network classifier |
| Scikit-learn | Preprocessing, feature scaling, dataset splitting, and evaluation |
| XGBoost | Attack classification |
| SHAP | Model explainability |
| Pandas | Dataset manipulation |
| NumPy | Numerical computation |
| Matplotlib | Model training and explainability visualizations |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive dashboard charts |
| Joblib | Saving and loading trained artifacts |

--- 
## 📊 Model Evaluation

The training pipeline generates classification reports for the evaluated models and the combined detection approach.

The primary evaluation measures include:

| Metric | What It Measures |
|---|---|
| Precision | How many predicted positive cases are actually positive |
| Recall | How many actual positive cases are detected |
| F1-score | Balance between precision and recall |
| Support | Number of actual samples in each class |


## 🎓 Key Learnings

This project provided practical experience with:

- Building and training neural-network-based Autoencoders.
- Applying reconstruction-error-based anomaly detection.
- Extracting latent representations for downstream machine-learning tasks.
- Training and evaluating XGBoost and neural-network classifiers.
- Exploring a hybrid anomaly detection and classification pipeline.
- Applying SHAP to investigate model predictions.
- Integrating machine-learning artifacts into a Streamlit application.
- Designing a workflow that connects data preprocessing, model training, evaluation, and interactive inference.

---
⭐ If you find this project interesting, consider starring the repository!
