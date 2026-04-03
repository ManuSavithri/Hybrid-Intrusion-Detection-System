import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def build_autoencoder(input_dim, encoding_dim=32):
    encoder_layer = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(encoding_dim, activation='relu', name='bottleneck')
    ], name='encoder')

    decoder_layer = Sequential([
        Dense(64, activation='relu', input_shape=(encoding_dim,)),
        Dense(input_dim, activation='sigmoid')
    ], name='decoder')

    inp = Input(shape=(input_dim,), name='ae_input')
    encoded = encoder_layer(inp)
    decoded = decoder_layer(encoded)
    autoencoder = Model(inputs=inp, outputs=decoded, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder_layer

def train_autoencoder(autoencoder, X_train_normal, epochs=100, batch_size=64):
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[es],
        verbose=1
    )

    plt.figure(figsize=(7,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("Autoencoder Loss (trained on normal samples)")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend()
    plt.show()
