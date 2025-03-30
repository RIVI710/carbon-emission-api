import pickle
from sklearn.ensemble import RandomForestRegressor

# ✅ Correct: Define the training data
X_train = [[1, 2], [3, 4], [5, 6]]
y_train = [10, 20, 30]

# ✅ Correct: Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# ✅ Correct: Save the model, NOT a NumPy array
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
