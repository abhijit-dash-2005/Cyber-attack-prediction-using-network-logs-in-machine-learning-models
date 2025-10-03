import requests
import joblib

# Load label encoder to decode returned class indices
label_encoder = joblib.load("label_encoder.pkl")

# URL of your running Flask API
url = "http://127.0.0.1:5000/predict"

# Replace below with actual feature names and values expected by your model
test_data = {
    "Flow Duration": 123,
    "Total Fwd Packets": 5,
    "Total Backward Packets": 7,
    # ... fill out all required features, exactly matching your training columns
}

# Send POST request
response = requests.post(url, json=test_data)
result = response.json()

# Extract and print human-readable label
predictions = result.get('predictions', [])
if predictions:
    print("Human-readable prediction:", predictions[0])
else:
    print("API response:", result)
