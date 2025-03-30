import pickle
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# ❌ Incorrect way (might be causing the issue)
model = np.load("model.pkl")  # This is incorrect because np.load() loads an array, not a model.

# ✅ Correct way to load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)  # This ensures it's a trained model, not an array.


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert features into a NumPy array
    features = np.array(data["features"]).reshape(1, -1)

    # ❌ If model is a NumPy array, `.predict()` will not work
    prediction = model.predict(features)  # Make sure model is a trained ML model

    return jsonify({"prediction": prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
