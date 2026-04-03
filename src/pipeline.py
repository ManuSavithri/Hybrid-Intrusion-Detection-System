from sklearn.metrics import classification_report

def combined_pipeline(anomaly_preds, X_test_latent, xgb_model, y_test_bin, y_test_type):
    combined_pred_bin, combined_pred_type = [], []
    for i in range(len(X_test_latent)):
        if anomaly_preds[i] == 0:
            combined_pred_bin.append(0)
            combined_pred_type.append(0)
        else:
            pred_type = xgb_model.predict(X_test_latent[i].reshape(1, -1))[0]
            combined_pred_bin.append(1)
            combined_pred_type.append(pred_type)

    print("=== Combined Pipeline - Binary Report ===")
    print(classification_report(y_test_bin, combined_pred_bin))
    print("=== Combined Pipeline - Multi-class Report ===")
    print(classification_report(y_test_type, combined_pred_type))
