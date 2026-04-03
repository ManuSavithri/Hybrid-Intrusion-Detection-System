# ===========================
# 0. Imports & reproducible seed
# ===========================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from xgboost import XGBClassifier
import shap

# ===========================
# Reproducibility
# ===========================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
random_state = SEED

# ===========================
# Example check (remove/replace later)
# ===========================
if __name__ == "__main__":
    print("✅ Environment ready with reproducibility set.")

    # Example plot to verify matplotlib works in VS Code
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, label="sin(x)")
    plt.legend()
    plt.title("Test Plot")
    plt.show()
