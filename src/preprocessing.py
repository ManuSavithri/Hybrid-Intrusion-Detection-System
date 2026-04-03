import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(data_path, random_state=42):
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)

    # Binary label: normal=0, attack=1
    df['label'] = df['Attack Type'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)
    y_attack_type = df['Attack Type'].astype(str).copy()

    drop_cols = ['Timestamp', 'Source IP Address', 'Destination IP Address',
                 'Attack Type', 'label']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y_binary = df['label'].values
    y_attack_type = y_attack_type.values

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].apply(lambda v: isinstance(v, str)).any():
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode targets
    y_binary_enc = LabelEncoder().fit_transform(y_binary.astype(str))
    y_attack_type_enc = LabelEncoder().fit_transform(y_attack_type.astype(str))

    # Train-test split
    X_train, X_test, y_train_bin, y_test_bin, y_train_type, y_test_type = train_test_split(
        X, y_binary_enc, y_attack_type_enc,
        test_size=0.2, random_state=random_state, stratify=y_binary_enc
    )

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled,
            y_train_bin, y_test_bin,
            y_train_type, y_test_type,
            scaler, X, y_attack_type_enc)
