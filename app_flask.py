from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

app = Flask(__name__)

# Load the trained models and preprocessing objects
try:
    rf_model = joblib.load('air_quality_model.pkl')
    xgb_model = joblib.load('optimized_air_quality_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("Models and preprocessing objects loaded successfully")
except Exception as e:
    print(f"Error loading models and preprocessing objects: {str(e)}")
    rf_model = None
    xgb_model = None
    scaler = None
    feature_names = None

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if rf_model is None or xgb_model is None or scaler is None or feature_names is None:
            return jsonify({'error': 'Models not loaded properly. Please check server logs.'}), 500

        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Create a DataFrame with the expected column names
        expected_columns = [
            'Temperature', 'Humidity', 'PM2.5', 'PM10',
            'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas',
            'Population_Density'
        ]
        
        # Check if all required fields are present
        missing_fields = [col for col in expected_columns if col not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        # Convert input data to DataFrame with correct column order
        input_data = pd.DataFrame([{col: data[col] for col in expected_columns}])
        
        # Scale the input data
        input_data_scaled = input_data.copy()
        input_data_scaled[feature_names] = scaler.transform(input_data[feature_names])
        
        # Make predictions using both models
        rf_prediction = rf_model.predict(input_data_scaled)[0]
        xgb_prediction = xgb_model.predict(input_data_scaled)[0]
        
        # Convert numeric predictions to labels
        quality_labels = {0: 'Good', 1: 'Moderate', 2: 'Poor', 3: 'Hazardous'}
        
        return jsonify({
            'random_forest_prediction': quality_labels[rf_prediction],
            'xgboost_prediction': quality_labels[xgb_prediction]
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 