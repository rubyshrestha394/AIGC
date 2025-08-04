from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, model_name, X_val, y_val, X_test, y_test):
    print(f"\n--- {model_name} Validation ---")
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)

    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Precision:", precision_score(y_val, y_val_pred))
    print("Recall:", recall_score(y_val, y_val_pred))
    print("F1-Score:", f1_score(y_val, y_val_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_val_prob))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.title(f"{model_name} - Confusion Matrix (Validation)")
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)  # Makes plots show up properly in some environments

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_val, y_val_prob):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    print(f"\n--- {model_name} Test ---")
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred))
    print("Recall:", recall_score(y_test, y_test_pred))
    print("F1-Score:", f1_score(y_test, y_test_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_test_prob))
