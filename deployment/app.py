# app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('models/traffic_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
le = pickle.load(open('models/label_encoder.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['temp'], data['rain_1h'], data['snow_1h'], data['clouds_all'],
                         le.transform([data['weather_main']])[0], le.transform([data['holiday']])[0]])
    
    features = scaler.transform([features])
    prediction = model.predict(features)
    
    return jsonify({'traffic_volume': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
