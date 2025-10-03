from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load saved model and preprocessing objects
rf_model = joblib.load("RandomForest_model_2.pkl")
# scaler = joblib.load("scaler.pkl")  # Uncomment if scaling was used
label_encoder = joblib.load("label_encoder.pkl")

# List your feature column names (order matters!)
FEATURE_COLUMNS = ['Flow Duration', 'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow IAT Std', 'Flow IAT Max', 'Fwd IAT Total', 'Fwd IAT Std', 'Fwd IAT Max', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'Avg Bwd Segment Size', 'Idle Mean', 'Idle Max', 'Idle Min']  # use your real list

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_json = request.get_json()
        if isinstance(input_json, list):
            input_df = pd.DataFrame(input_json)
        else:
            input_df = pd.DataFrame([input_json])
        input_df = input_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        # input_scaled = scaler.transform(input_df)  # Uncomment if you used scaling
        input_scaled = input_df  # Use directly if you did not use scaling

        pred = rf_model.predict(input_scaled)
        decoded_label = label_encoder.inverse_transform(pred)
        return jsonify({"predictions": decoded_label.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
