import numpy as np
from sklearn.metrics import f1_score

def compute_mse_on(autoenc, X):
    recons = autoenc.predict(X, verbose=0)
    return np.mean(np.power(X - recons, 2), axis=1)

def find_best_threshold(X_val_norm, X_val_anom, autoenc):
    X_val = np.vstack([X_val_norm, X_val_anom])
    y_val = np.hstack([np.zeros(len(X_val_norm)), np.ones(len(X_val_anom))])
    mse_vals = compute_mse_on(autoenc, X_val)

    perc_vals = np.percentile(mse_vals, np.linspace(50, 99.9, 200))
    best_f1, best_t = -1, None
    for t in perc_vals:
        preds = (mse_vals > t).astype(int)
        if len(np.unique(preds)) > 1:
            f1 = f1_score(y_val, preds)
            if f1 > best_f1:
                best_f1, best_t = f1, t
    return best_t, best_f1

def robust_threshold(autoenc, X_train_scaled, y_train_bin, n_repeats=10, val_norm_frac=0.2, val_anom_frac=0.2):
    thresholds, f1s = [], []
    norm_idx = np.where(y_train_bin == 0)[0]
    anom_idx = np.where(y_train_bin == 1)[0]

    if len(anom_idx) == 0:
        train_mse = compute_mse_on(autoenc, X_train_scaled)
        return float(np.percentile(train_mse, 99)), None, [], []

    for _ in range(n_repeats):
        np.random.shuffle(norm_idx); np.random.shuffle(anom_idx)
        n_val_norm = int(len(norm_idx) * val_norm_frac)
        n_val_anom = int(len(anom_idx) * val_anom_frac)
        val_norm, val_anom = norm_idx[:n_val_norm], anom_idx[:n_val_anom]

        if len(val_norm) > 0 and len(val_anom) > 0:
            t, f1 = find_best_threshold(X_train_scaled[val_norm], X_train_scaled[val_anom], autoenc)
            if t: thresholds.append(t); f1s.append(f1)

    median_t = float(np.median(thresholds))
    median_f1 = float(np.median(f1s)) if f1s else None
    return median_t, median_f1, thresholds, f1s
