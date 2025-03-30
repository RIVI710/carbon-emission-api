import requests

url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}
data = {"features": [1, 2]}

response = requests.post(url, json=data, headers=headers)

print("Response:", response.json())
