"""
YuvaIntern - Telecom Data Analytics Engineer Internship
Week 2: Data Preprocessing and Feature Engineering Pipeline
Author: Abdul Soleh
Repository: telecom-customer-churn-analytics
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def generate_synthetic_telecom_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Simulates raw telecom dataset with typical industry defects:
    whitespace blanks, missing entries, duplicate rows, and skewed outliers.
    """
    np.random.seed(random_state)
    
    customer_ids = [f"TEL-{1000 + i}" for i in range(n_samples)]
    tenure = np.random.randint(1, 73, size=n_samples)
    monthly_charges = np.round(np.random.uniform(18.5, 118.5, size=n_samples), 2)
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    tech_support = np.random.choice(["Yes", "No", "No internet"], size=n_samples, p=[0.35, 0.45, 0.20])
    online_sec = np.random.choice(["Yes", "No", "No internet"], size=n_samples, p=[0.30, 0.50, 0.20])
    paperless_bill = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    service_calls = np.random.poisson(lam=1.7, size=n_samples)
    data_usage_gb = np.random.exponential(scale=24.0, size=n_samples)
    
    total_charges = tenure * monthly_charges + np.random.normal(0, 3, size=n_samples)
    total_charges = np.round(total_charges, 2)
    
    # Calculate churn indicator
    churn_prob = (
        (contracts == "Month-to-month") * 0.35 +
        (service_calls >= 4) * 0.35 +
        (monthly_charges > 80.0) * 0.15 -
        (tenure / 72.0) * 0.30
    )
    churn = np.where(1 / (1 + np.exp(-churn_prob)) > 0.52, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_ids,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges.astype(object),
        "Contract": contracts,
        "TechSupport": tech_support,
        "OnlineSecurity": online_sec,
        "PaperlessBilling": paperless_bill,
        "CustServCalls": service_calls,
        "TotalDataUsageGB": np.round(data_usage_gb, 2),
        "Churn": churn
    })

    # Inject real-world defects
    # 1. Whitespace blanks in TotalCharges (common in IBM Telco dataset for tenure=1)
    df.loc[df["tenure"] == 1, "TotalCharges"] = " "

    # 2. Duplicate records
    duplicates = df.iloc[:15].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # 3. Extreme outliers in network data usage
    df.loc[20:23, "TotalDataUsageGB"] = [480.5, 530.2, 612.0, 495.8]

    return df


def clean_telecom_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans duplicates, resolves whitespace/null entries, and caps outliers using IQR.
    """
    df_cleaned = df.copy()

    # Step 1: Remove duplicate records
    initial_count = len(df_cleaned)
    df_cleaned.drop_duplicates(subset=["customerID"], keep="first", inplace=True)
    df_cleaned.reset_index(drop=True, inplace=True)
    print(f"[*] Deduping: Removed {initial_count - len(df_cleaned)} duplicate rows.")

    # Step 2: Handle whitespace strings in TotalCharges & impute
    df_cleaned["TotalCharges"] = pd.to_numeric(
        df_cleaned["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    # New customers with missing TotalCharges are imputed with their first month charge
    df_cleaned["TotalCharges"] = df_cleaned["TotalCharges"].fillna(df_cleaned["MonthlyCharges"])
    print(f"[*] Missing Values Handled: TotalCharges null count = {df_cleaned['TotalCharges'].isnull().sum()}")

    # Step 3: Outlier Capping via IQR for skewed usage metrics
    q25 = df_cleaned["TotalDataUsageGB"].quantile(0.25)
    q75 = df_cleaned["TotalDataUsageGB"].quantile(0.75)
    iqr = q75 - q25
    upper_threshold = q75 + (1.5 * iqr)
    df_cleaned["TotalDataUsageGB"] = np.where(
        df_cleaned["TotalDataUsageGB"] > upper_threshold, 
        upper_threshold, 
        df_cleaned["TotalDataUsageGB"]
    )
    print(f"[*] Outlier Capping: TotalDataUsageGB capped at {upper_threshold:.2f} GB")

    return df_cleaned


def engineer_telecom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts high-impact domain features from raw telecom columns.
    """
    df_feat = df.copy()

    # 1. Consumption Intensity: Average Monthly Data Usage
    df_feat["AvgMonthlyDataGB"] = np.round(df_feat["TotalDataUsageGB"] / df_feat["tenure"], 2)

    # 2. Financial Velocity: Charge per Tenure Ratio
    df_feat["ChargePerTenureRatio"] = np.round(df_feat["MonthlyCharges"] / (df_feat["tenure"] + 1), 3)

    # 3. Customer Dissatisfaction: Escalation Trigger Flag (>= 4 calls)
    df_feat["HighSupportCallFlag"] = (df_feat["CustServCalls"] >= 4).astype(int)

    # 4. Service Stickiness: Total Value-Added Services (VAS) count
    vas_columns = ["TechSupport", "OnlineSecurity"]
    df_feat["TotalVASCount"] = df_feat[vas_columns].apply(lambda row: sum(row == "Yes"), axis=1)

    print("[*] Feature Engineering: Created AvgMonthlyDataGB, ChargePerTenureRatio, HighSupportCallFlag, TotalVASCount.")
    return df_feat


def build_and_apply_preprocessor(df: pd.DataFrame):
    """
    Constructs ColumnTransformer pipeline with RobustScaler and OneHotEncoder.
    """
    numeric_cols = [
        "tenure", "MonthlyCharges", "TotalCharges", 
        "AvgMonthlyDataGB", "ChargePerTenureRatio", "TotalDataUsageGB"
    ]
    categorical_cols = ["Contract", "TechSupport", "OnlineSecurity", "PaperlessBilling"]
    passthrough_cols = ["HighSupportCallFlag", "TotalVASCount"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
            ("pass", "passthrough", passthrough_cols)
        ]
    )

    X = df[numeric_cols + categorical_cols + passthrough_cols]
    y = (df["Churn"] == "Yes").astype(int)

    X_transformed = preprocessor.fit_transform(X)
    return preprocessor, X_transformed, y


if __name__ == "__main__":
    print("=" * 65)
    print(" YuvaIntern Telecom Preprocessing & Feature Engineering Pipeline ")
    print("=" * 65)

    # 1. Generate / Ingest Data
    raw_data = generate_synthetic_telecom_data(n_samples=1000)
    print(f"[+] Loaded raw dataset: {raw_data.shape[0]} rows, {raw_data.shape[1]} columns")

    # 2. Clean Data
    clean_data = clean_telecom_data(raw_data)

    # 3. Engineer Domain Features
    feature_data = engineer_telecom_features(clean_data)

    # 4. Transform & Scale
    preprocessor, X_final, y_final = build_and_apply_preprocessor(feature_data)

    print("-" * 65)
    print(f"[✓] Transformed Feature Matrix Shape: {X_final.shape}")
    print(f"[✓] Target Array Shape: {y_final.shape} (Churn Rate: {y_final.mean():.2%})")
    print(f"[✓] Missing Values in Processed Matrix: {np.isnan(X_final).sum()}")
    print("=" * 65)
    print(">> Pipeline execution successful. Dataset ready for Week 3 modeling.")
