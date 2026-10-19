"""
Telecom Model Evaluation and Error Analysis
Week 4 Task: Metric Calculation, Validation, and Diagnostic Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def generate_mock_telecom_data(n_samples: int = 5000, random_state: int = 42):
    """Generates synthetic telecom churn data with realistic imbalance (25% churn)."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=3,
        weights=[0.75, 0.25],
        random_state=random_state
    )
    feature_names = [
        "Tenure_Months", "Monthly_Charges", "Total_Charges", 
        "Support_Tickets", "Contract_Type", "Payment_Method",
        "Internet_Service", "Paperless_Billing", "Streaming_Addons", "Senior_Citizen"
    ]
    df = pd.DataFrame(X, columns=feature_names)
    return df, pd.Series(y, name="Churn")


def evaluate_telecom_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Performs Stratified 5-Fold Cross Validation and generates
    telecom-specific performance metrics and error distributions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

    fold_metrics = []
    oof_predictions = np.zeros(len(y))
    oof_probabilities = np.zeros(len(y))

    print(f"--- Running {n_splits}-Fold Stratified Cross-Validation ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        oof_predictions[val_idx] = preds
        oof_probabilities[val_idx] = probs

        fold_metrics.append({
            "Fold": fold,
            "Accuracy": accuracy_score(y_val, preds),
            "Precision": precision_score(y_val, preds),
            "Recall": recall_score(y_val, preds),
            "F1": f1_score(y_val, preds),
            "ROC_AUC": roc_auc_score(y_val, probs)
        })

    metrics_df = pd.DataFrame(fold_metrics)
    print("\nCross-Validation Summary:")
    print(metrics_df.mean(numeric_only=True).round(4))

    # Error Analysis via Out-Of-Fold Predictions
    cm = confusion_matrix(y, oof_predictions)
    tn, fp, fn, tp = cm.ravel()

    print("\n--- Out-of-Fold Confusion Matrix ---")
    print(f"True Negatives  (Loyal Retained)      : {tn}")
    print(f"False Positives (Wasted Incentive)    : {fp}")
    print(f"False Negatives (Uncaught Churn)      : {fn}")
    print(f"True Positives  (Successfully Caught) : {tp}")

    # Cost-benefit calculation example
    c_retention = 15.0   # Cost of discount/voucher for FP
    c_churn_loss = 120.0 # Lost Customer Lifetime Value for FN
    total_cost = (fp * c_retention) + (fn * c_churn_loss)
    print(f"\nEstimated Financial Loss from Errors: ${total_cost:,.2f}")

    return metrics_df, oof_predictions, oof_probabilities


if __name__ == "__main__":
    X_data, y_labels = generate_mock_telecom_data()
    metrics, oof_preds, oof_probs = evaluate_telecom_model(X_data, y_labels)
