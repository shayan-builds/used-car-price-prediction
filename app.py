from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
CORS(app)

model = joblib.load('car_price_model.pkl')
with open('columns.json') as f:
    columns = json.load(f)['data_columns']

def build_input_row(data):
    row = pd.Series(0, index=columns)
    row['vehicle_age'] = data['vehicle_age']
    row['km_driven'] = data['km_driven']
    row['mileage'] = data['mileage']
    row['engine'] = data['engine']
    row['max_power'] = data['max_power']
    row['seats'] = data['seats']

    brand_col = f"brand_{data['brand']}"
    if brand_col in columns:
        row[brand_col] = 1

    fuel_col = f"fuel_type_{data['fuel_type']}"
    if fuel_col in columns:
        row[fuel_col] = 1

    seller_col = f"seller_type_{data['seller_type']}"
    if seller_col in columns:
        row[seller_col] = 1

    trans_col = f"transmission_type_{data['transmission_type']}"
    if trans_col in columns:
        row[trans_col] = 1

    return row

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    row = build_input_row(data)
    price = model.predict([row.values])[0]
    return jsonify({'predicted_price': round(float(price), 2)})

@app.route('/price-range', methods=['POST'])
def price_range():
    data = request.json
    row = build_input_row(data)
    tree_preds = np.array([tree.predict([row.values])[0] for tree in model.estimators_])
    return jsonify({
        'lower': round(float(np.percentile(tree_preds, 10)), 2),
        'mean': round(float(tree_preds.mean()), 2),
        'upper': round(float(np.percentile(tree_preds, 90)), 2)
    })

@app.route('/depreciation', methods=['POST'])
def depreciation():
    data = request.json
    results = []
    for age in range(0, 11):
        row = build_input_row(data)
        row['vehicle_age'] = age
        price = model.predict([row.values])[0]
        results.append({'year': age, 'predicted_price': round(float(price), 2)})
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)