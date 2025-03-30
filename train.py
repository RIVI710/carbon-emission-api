import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ✅ Load dataset
df = pd.read_csv("cleaned_data.csv")

# ✅ Define correct numeric features
features = [
    "2017 NAICS Code",
    "Supply Chain Emission Factors without Margins",
    "Margins of Supply Chain Emission Factors",
    "Supply Chain Emission Factors with Margins"
]

# ✅ Select features and target
X = df[features]
y = df["Supply Chain Emission Factors with Margins"]  # 🔍 Choose correct target

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# ✅ Save model
import joblib
joblib.dump(model, "model.pkl")

print("✅ Model trained successfully!")
print(f"Model expects {X_train.shape[1]} features.")
