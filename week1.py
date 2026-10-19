
---

### 2. `data_simulation_week1.py` (डेटा सिमुलेशन स्क्रिप्ट)
*टास्क में लिखा है: "suggest methods for simulating additional data if required using Python". यह Python स्क्रिप्ट रिपॉजिटरी को टेक्निकल बनाती है:*

```python
"""
Telecom Customer Data Simulation Script (Week 1 Task)
Author: Abdul Soleh
Role: Telecom Data Analytics Engineer Intern
"""

import numpy as np
import pandas as pd

def simulate_telecom_data(n_samples=500, random_seed=42):
    """
    Simulates telecom customer demographic, network usage, and churn labels.
    """
    np.random.seed(random_seed)
    
    customer_ids = [f"TELCO-{1000 + i}" for i in range(n_samples)]
    tenure_months = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2)
    contract_types = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    data_usage_gb = np.round(np.random.gamma(shape=3.0, scale=8.0, size=n_samples), 2)
    customer_service_calls = np.random.poisson(lam=1.5, size=n_samples)
    
    # Churn probability logic based on domain factors
    churn_score = (
        (contract_types == "Month-to-month") * 0.35 +
        (customer_service_calls > 3) * 0.40 +
        (monthly_charges > 85.0) * 0.20 -
        (tenure_months / 72.0) * 0.30
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churn = np.where(churn_prob > 0.55, "Yes", "No")
    
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "TenureMonths": tenure_months,
        "MonthlyCharges": monthly_charges,
        "Contract": contract_types,
        "DataUsageGB": data_usage_gb,
        "CustomerServiceCalls": customer_service_calls,
        "Churn": churn
    })
    
    return df

if __name__ == "__main__":
    synthetic_df = simulate_telecom_data(500)
    print("--- Telecom Synthetic Data Preview (First 5 records) ---")
    print(synthetic_df.head())
    print("\nSimulated Churn Distribution:")
    print(synthetic_df["Churn"].value_counts(normalize=True))