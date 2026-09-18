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
    row = pd.Series(0.0, index=columns, dtype=float)

    # Safe numeric assignments with fallback defaults
    for num_field, default_val in [
        ('vehicle_age', 5),
        ('km_driven', 40000),
        ('mileage', 18.0),
        ('engine', 1200),
        ('max_power', 85.0),
        ('seats', 5)
    ]:
        val = data.get(num_field, default_val)
        try:
            row[num_field] = float(val) if val is not None else float(default_val)
        except (ValueError, TypeError):
            row[num_field] = float(default_val)

    # Extract categorical inputs safely, supporting multiple possible key names
    brand = str(data.get('brand', '')).strip()
    fuel = str(data.get('fuel_type', data.get('fuel', ''))).strip()
    seller = str(data.get('seller_type', data.get('seller', ''))).strip()
    trans = str(data.get('transmission_type', data.get('transmission', ''))).strip()

    # Case-insensitive column matching
    cols_lookup = {c.lower(): c for c in columns}

    for prefix, val in [
        ('brand_', brand),
        ('fuel_type_', fuel),
        ('seller_type_', seller),
        ('transmission_type_', trans),
        ('transmission_', trans),
        ('model_', str(data.get('model', '')).strip())
    ]:
        if val:
            candidate = f"{prefix}{val}".lower()
            if candidate in cols_lookup:
                row[cols_lookup[candidate]] = 1.0

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