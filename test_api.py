import requests

url = "http://127.0.0.1:5000/predict"
data = {"features": [111110, 1.326, 0.103, 1.326]}  # ✅ 4 features

response = requests.post(url, json=data)
print(response.json())  # Expected: {'prediction': some_value}
