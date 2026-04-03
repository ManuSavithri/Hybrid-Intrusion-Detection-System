import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

def train_xgb(X_train, y_train, X_test, y_test, desc=""):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"=== XGBoost {desc} Classification Report ===")
    print(classification_report(y_test, preds))
    return model, preds

def train_nn(X_train, y_train, X_test, y_test, num_classes):
    clf = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    clf.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)], verbose=0)
    preds = np.argmax(clf.predict(X_test), axis=1)
    print("=== Dense NN Classification Report ===")
    print(classification_report(y_test, preds))
    return clf, preds
