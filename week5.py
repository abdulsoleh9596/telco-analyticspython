"""
Telecom Churn Prediction & Business Insights Pipeline
Week 5 Capstone - Telecom Analytics & Reporting
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def generate_synthetic_telecom_data(n_samples: int = 5000, random_state: int = 42):
    """Generates realistic synthetic telecom operational data."""
    np.random.seed(random_state)

    tenure_months = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20])
    packet_loss_pct = np.random.exponential(scale=1.2, size=n_samples)
    support_tickets = np.random.poisson(lam=1.5, size=n_samples)
    usage_delta_30_90 = np.random.normal(loc=0.0, scale=0.15, size=n_samples)

    # Churn probability calculation based on operational drivers
    linear_risk = (
        -1.8
        - 0.04 * tenure_months
        + 0.025 * monthly_charges
        - 0.9 * contract_type
        + 0.35 * packet_loss_pct
        + 0.45 * support_tickets
        - 1.5 * usage_delta_30_90
    )
    prob_churn = 1 / (1 + np.exp(-linear_risk))
    churn = (np.random.rand(n_samples) < prob_churn).astype(int)

    df = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "contract_type": contract_type,
            "packet_loss_pct": packet_loss_pct,
            "support_tickets": support_tickets,
            "usage_delta_30_90": usage_delta_30_90,
            "churn": churn,
        }
    )
    return df


def train_and_evaluate(df: pd.DataFrame):
    """Executes model training, evaluation, and metric calculation."""
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=scale_pos,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    results = {}
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds_proba = model.predict_proba(X_test_scaled)[:, 1]
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds_proba = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)

        auc = roc_auc_score(y_test, preds_proba)
        f1 = f1_score(y_test, preds)
        fpr, tpr, _ = roc_curve(y_test, preds_proba)

        results[name] = {"ROC-AUC": round(auc, 3), "F1-Score": round(f1, 3)}
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.title("ROC Curve Benchmark - Telecom Churn Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("telecom_churn_roc_curve.png", dpi=300)
    plt.close()

    print("Model Evaluation Summary:")
    print(pd.DataFrame(results).T)

    # Output feature importance from champion model
    xgb_model = models["XGBoost"]
    importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("\nChampion Model (XGBoost) Feature Importances:")
    print(importances)


if __name__ == "__main__":
    data = generate_synthetic_telecom_data()
    train_and_evaluate(data)
