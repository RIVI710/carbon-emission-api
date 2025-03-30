from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load trained model
model = joblib.load("model.pkl")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # ✅ Ensure correct feature input
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key in request."}), 400

    features = data["features"]

    if len(features) != 4:  # 🚀 Expecting 4 features now
        return jsonify({"error": f"Model expects 4 features, but got {len(features)}."}), 400

    # ✅ Convert to NumPy array and reshape
    features = np.array(features).reshape(1, -1)

    # ✅ Predict
    prediction = model.predict(features)[0]

    return jsonify({"prediction": prediction})


if __name__ == '__main__':
    app.run(debug=True)
