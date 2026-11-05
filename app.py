import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("======================================")
print(" CREDIT CARD FRAUD DETECTION SYSTEM")
print("======================================")

# Generate Sample Dataset
np.random.seed(42)

data = pd.DataFrame({
    "amount": np.random.randint(100, 50000, 1000),
    "transaction_hour": np.random.randint(0, 24, 1000),
    "distance": np.random.randint(1, 1000, 1000)
})

# Create Fraud Labels
data["fraud"] = (
    (data["amount"] > 30000) &
    (data["transaction_hour"] < 5) &
    (data["distance"] > 500)
).astype(int)

# Features
X = data[["amount", "transaction_hour", "distance"]]

# Target
y = data["fraud"]

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("\nMachine Learning Model Trained Successfully!")

# User Input
print("\nEnter Transaction Details")

amount = float(input("Transaction Amount: "))
hour = int(input("Transaction Hour (0-23): "))
distance = float(input("Distance from Last Transaction (km): "))

input_data = pd.DataFrame([{
    "amount": amount,
    "transaction_hour": hour,
    "distance": distance
}])

# Prediction
prediction = model.predict(input_data)

print("\n======================================")
print(" TRANSACTION ANALYSIS RESULT")
print("======================================")

if prediction[0] == 1:
    print("Transaction Status: FRAUDULENT")
    print("Risk Level: HIGH")
    print("Recommendation: Verify the transaction.")

else:
    print("Transaction Status: LEGITIMATE")
    print("Risk Level: LOW")
    print("Recommendation: Transaction appears normal.")

print("======================================")