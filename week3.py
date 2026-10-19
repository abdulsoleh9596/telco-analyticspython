"""
YuvaIntern - Telecom Data Analytics Engineer Internship
Week 3: Predictive Model Training, Hyperparameter Tuning & Cross-Validation
Author: Abdul Soleh
Repository: telecom-customer-churn-analytics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def generate_and_preprocess_data(n_samples: int = 1500, random_state: int = 42):
    """
    Generates synthetic telecom records and applies the Week 2 preprocessing pipeline.
    """
    np.random.seed(random_state)
    customer_ids = [f"TEL-{1000 + i}" for i in range(n_samples)]
    tenure = np.random.randint(1, 73, size=n_samples)
    monthly_charges = np.round(np.random.uniform(18.5, 118.5, size=n_samples), 2)
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    tech_support = np.random.choice(["Yes", "No", "No internet"], size=n_samples, p=[0.35, 0.45, 0.20])
    online_sec = np.random.choice(["Yes", "No", "No internet"], size=n_samples, p=[0.30, 0.50, 0.20])
    service_calls = np.random.poisson(lam=1.8, size=n_samples)
    data_usage_gb = np.random.exponential(scale=25.0, size=n_samples)
    
    total_charges = tenure * monthly_charges + np.random.normal(0, 3, size=n_samples)
    total_charges = np.round(total_charges, 2)
    
    # Ground-truth churn calculation with realistic domain weights
    churn_prob = (
        (contracts == "Month-to-month") * 0.40 +
        (service_calls >= 4) * 0.45 +
        (monthly_charges > 85.0) * 0.20 -
        (tenure / 72.0) * 0.35
    )
    churn = np.where(1 / (1 + np.exp(-churn_prob)) > 0.50, 1, 0)

    df = pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contracts,
        "TechSupport": tech_support,
        "OnlineSecurity": online_sec,
        "CustServCalls": service_calls,
        "TotalDataUsageGB": data_usage_gb,
        "Churn": churn
    })

    # Domain feature engineering
    df["AvgMonthlyDataGB"] = np.round(df["TotalDataUsageGB"] / df["tenure"], 2)
    df["ChargePerTenureRatio"] = np.round(df["MonthlyCharges"] / (df["tenure"] + 1), 3)
    df["HighSupportCallFlag"] = (df["CustServCalls"] >= 4).astype(int)
    df["TotalVASCount"] = (df["TechSupport"] == "Yes").astype(int) + (df["OnlineSecurity"] == "Yes").astype(int)

    # Scikit-Learn Preprocessing Pipeline
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlyDataGB", "ChargePerTenureRatio", "TotalDataUsageGB"]
    cat_cols = ["Contract", "TechSupport", "OnlineSecurity"]
    pass_cols = ["HighSupportCallFlag", "TotalVASCount"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
            ("pass", "passthrough", pass_cols)
        ]
    )

    X = preprocessor.fit_transform(df[num_cols + cat_cols + pass_cols])
    y = df["Churn"].values
    return X, y


def train_and_evaluate_models(X, y):
    """
    Executes stratified splitting, candidate benchmarking, and GridSearchCV tuning.
    """
    print("=" * 65)
    print(" YuvaIntern Telecom Churn Predictive Modeling - Week 3")
    print("=" * 65)

    # Step 1: Stratified Data Partitioning (70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"[*] Training samples: {X_train.shape[0]}, Test holdout samples: {X_test.shape[0]}")
    print(f"[*] Train Churn Rate: {y_train.mean():.2%}, Test Churn Rate: {y_test.mean():.2%}")

    # Step 2: Baseline Candidate Evaluation with 5-Fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest (Default)": RandomForestClassifier(class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    print("\n--- 5-Fold Stratified Cross-Validation Benchmark ---")
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"[*] {name:25s} | Mean CV ROC-AUC: {scores.mean():.4f} (±{scores.std():.4f})")

    # Step 3: Hyperparameter Optimization for Random Forest via GridSearchCV
    print("\n--- Hyperparameter Tuning: Random Forest Classifier ---")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 8, 12],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print(f"[+] Best Hyperparameters Found: {grid_search.best_params_}")
    print(f"[+] Best Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}")

    # Step 4: Final Holdout Test Evaluation
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    test_recall = recall_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print(" Final Holdout Test Evaluation (Best Random Forest Model) ")
    print("=" * 65)
    print(f"[✓] Test ROC-AUC Score: {test_roc_auc:.4f}")
    print(f"[✓] Churn Class Recall:  {test_recall:.2%}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Retained (0)", "Churned (1)"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("=" * 65)
    print(">> Model training complete. Pipeline ready for Week 4 performance analysis.")


if __name__ == "__main__":
    X_matrix, y_vector = generate_and_preprocess_data(n_samples=1500)
    train_and_evaluate_models(X_matrix, y_vector)
