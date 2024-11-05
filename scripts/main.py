import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('../src'))

from data_loader import load_data, load_datasets, rename_columns
from analysis import calculate_moving_averages, merge_datasets
from visualization import plot_price_trend, plot_moving_averages, plot_prices, plot_with_annotation, plot_residuals, plot_forecast, plot_actual_vs_predicted

from feature_engineering import add_time_features, split_data, generate_future_dates, create_future_features, forecast_future
from model_training import initialize_xgboost, train_model, evaluate_model, display_metrics

def main(oil_path, gas_path):

    data = load_data(oil_path)

    # Plot original price trend
    plot_price_trend(data)

    # Calculate and plot moving averages
    data = calculate_moving_averages(data, windows=[3, 6, 12])
    plot_moving_averages(data)

    # Load data
    oil_data = load_datasets(oil_path)
    gas_data = load_datasets(gas_path)

    # Rename columns
    oil_data = rename_columns(oil_data, {'Price': 'Oil_price'})
    gas_data = rename_columns(gas_data, {'Price': 'Gas_price'})

    # Merge datasets
    merged_data = merge_datasets(oil_data, gas_data)

    # Plot prices and moving averages
    plot_prices(merged_data)
    plot_with_annotation(merged_data)

    # Feature Engineering
    data = add_time_features(data)
    X_train, X_test, y_train, y_test = split_data(data, target_column='Price')

    # Model Training
    model = initialize_xgboost()
    model = train_model(model, X_train, y_train, X_test, y_test)

    # Model Evaluation
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    display_metrics('XGBoost', metrics)

    # Residuals and Forecasting
    residuals = y_test - y_pred
    plot_residuals(y_pred, residuals)

    # Forecast Future Values
    last_date = data.index[-1]
    future_dates = generate_future_dates(last_date)
    future_df = create_future_features(future_dates)
    future_predictions = forecast_future(model, future_df)
    plot_forecast(data, future_dates, future_predictions)

    # Plot Actual vs Predicted
    plot_actual_vs_predicted(y_test, y_pred)



if __name__ == "__main__":
    oil_path = '../data/Brent_Oil_Prices.csv'
    gas_path = '../data/natural_gas_daily.csv'
    main(oil_path, gas_path)
