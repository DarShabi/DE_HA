from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        data = pd.DataFrame(json_data)

        # Extract features and true labels
        true_labels = data['tnufa_endreasonName_mapped']
        features = data.drop(columns=['tnufa_endreasonName_mapped'])

        # Make predictions
        predicted_labels = model.predict(features)

        # Return both true labels and predicted labels in the response
        result = {
            "true_labels": true_labels.tolist(),
            "predicted_labels": predicted_labels.tolist()
        }

        return jsonify(result)  # Return predictions and true labels as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
