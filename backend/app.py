from flask import Flask, jsonify, request
import pandas as pd
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Load data
data = pd.read_csv('../data/Brent_Oil_Prices.csv')
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce') 
data.set_index('Date', inplace=True)

# Routes
@app.route('/')
def home():
    return "Welcome"

@app.route('/api/prices', methods=['GET'])
def get_prices():
    # Convert data to JSON format
    prices = data.reset_index().to_dict(orient='records')
    return jsonify(prices)

@app.route('/api/prices/range', methods=['GET'])
def get_price_range():
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    # Filter data based on the date range
    filtered_data = data.loc[start_date:end_date].reset_index().to_dict(orient='records')
    return jsonify(filtered_data)

if __name__ == '__main__':
    app.run(debug=True)
