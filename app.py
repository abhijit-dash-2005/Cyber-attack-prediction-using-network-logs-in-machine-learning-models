from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load your saved model and preprocessing objects
rf_model = joblib.load("RandomForest_model_3.pkl")
# scaler = joblib.load("scaler.pkl")  # Uncomment if you used scaling
label_encoder = joblib.load("label_encoder.pkl")

# Feature columns (order matters)
FEATURE_COLUMNS = [
    'Flow Duration', 'Bwd Packet Length Max', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow IAT Std', 'Flow IAT Max',
    'Fwd IAT Total', 'Fwd IAT Std', 'Fwd IAT Max', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'Avg Bwd Segment Size', 'Idle Mean', 'Idle Max', 'Idle Min'
]

app = Flask(__name__)

# Home route for status check (fixes 404 issue at root)
@app.route('/', methods=['GET'])
def home():
    return "Cyber Attack Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_json = request.get_json()
        if isinstance(input_json, list):
            input_df = pd.DataFrame(input_json)
        else:
            input_df = pd.DataFrame([input_json])
        input_df = input_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        # If you used scaling during training, uncomment below:
        # input_scaled = scaler.transform(input_df)
        input_scaled = input_df  # Use raw input if you did not scale the data
        pred = rf_model.predict(input_scaled)
        decoded_label = label_encoder.inverse_transform(pred)
        return jsonify({"predictions": decoded_label.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
