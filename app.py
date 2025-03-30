import os
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return "Carbon Emission API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [data["features"]]
    prediction = model.predict(features).tolist()
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's PORT or fallback
    app.run(host="0.0.0.0", port=port)
