import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    df_fraud = pd.read_csv('CreditCardData.csv', on_bad_lines='skip')
    print(f"Successfully loaded {len(df_fraud)} rows.")
except FileNotFoundError:
    print("Error: Please upload 'CreditCardData.csv' to the Colab file sidebar.")

X_fraud = df_fraud.select_dtypes(include=[np.number])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fraud)

model = IsolationForest(contamination=0.01, random_state=42)
df_fraud['anomaly_score'] = model.fit_predict(X_scaled)

fraud_detected = list(df_fraud['anomaly_score']).count(-1)

print("\n--- Task 3: Fraud Detection Report ---")
print(f"Total Transactions Analyzed: {len(df_fraud)}")
print(f"Fraudulent Transactions Detected: {fraud_detected}")

# Optional: View the first few detected fraudulent rows
print("\nSample of Detected Fraudulent Transactions:")
print(df_fraud[df_fraud['anomaly_score'] == -1].head())