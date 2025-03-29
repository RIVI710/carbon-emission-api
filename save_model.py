import joblib
from sklearn.ensemble import RandomForestRegressor

# Load or Train Your Model (if not already trained)
# Replace with your actual trained model
model = RandomForestRegressor(random_state=42)  # Example, use your trained model

# Save the model
joblib.dump(model, 'carbon_emission_model.pkl')

print("Model saved successfully!")
