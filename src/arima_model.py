# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# Checking for stationarity using ADF Test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')
        

def plot_differenced_price(data):

    # Differencing to achieve stationarity (if needed)
    data_diff = data['Price'].diff().dropna()
    plt.figure(figsize=(12,6))
    plt.plot(data_diff, label='Differenced Price')
    plt.title('Differenced Brent Oil Price')
    plt.xlabel('Date')
    plt.ylabel('Differenced Price')
    plt.legend()
    plt.savefig('../figures/differenced_price.png', format='png', dpi=300)
    plt.show()

    # Rechecking stationarity after differencing
    print("ADF Test Results for Differenced Price:")
    adf_test(data_diff)

    # Plotting ACF and PACF to determine ARIMA parameters
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(data_diff, ax=ax[0])
    plot_pacf(data_diff, ax=ax[1])
    plt.savefig('../figures/acf_pcaf_plots.png', format='png', dpi=300)
    plt.show()
def arima_model(data):

    # Splitting data into training and testing sets
    train_size = int(len(data) * 0.8)
    train, test = data['Price'][:train_size], data['Price'][train_size:]

    # Building the ARIMA model (ARIMA(p,d,q))
    # Assuming from ACF and PACF plots p=1, d=1, q=1 as a starting point; can tune further if needed.
    model = ARIMA(train, order=(1, 1, 1))
    arima_model = model.fit()

    # Model summary
    print(arima_model.summary())

    # Forecasting on the test set
    forecast = arima_model.forecast(steps=len(test))
    forecast_index = test.index

    # Plotting actual vs. predicted prices
    plt.figure(figsize=(12,6))
    plt.plot(train, label='Training Data')
    plt.plot(test, label='Test Data', color='blue')
    plt.plot(forecast_index, forecast, label='Forecast', color='orange')
    plt.title('Brent Oil Price Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('../figures/ARIMA_forcat_vs_actual.png', format='png', dpi=300)
    plt.show()

    # Calculating the performance metrics
    mse = mean_squared_error(test, forecast)
    print(f"Mean Squared Error: {mse:.2f}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Forecasting future prices (e.g., next 30 days)
    future_forecast = arima_model.forecast(steps=30)
    future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')

    plt.figure(figsize=(12,6))
    plt.plot(data['Price'], label='Historical Price')
    plt.plot(future_index, future_forecast, label='Future Forecast', color='red')
    plt.title('Brent Oil Price Future Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('../figures/ARIMA_future_forcast.png', format='png', dpi=300)
    plt.show()

def sarima_model(data):

    price_data = data['Price']

    # Differencing to make the series stationary (if necessary)
    price_data_diff = price_data.diff().dropna()

    # Perform grid search using auto_arima
    model = auto_arima(
        price_data,                # Original series
        start_p=0, max_p=5,        # Range of p values
        start_d=0, max_d=2,        # Range of d values
        start_q=0, max_q=5,        # Range of q values
        seasonal=False,            # Set to True if using SARIMA (seasonal ARIMA)
        trace=True,                # Show the search process
        error_action='ignore',     # Ignore errors during the search
        suppress_warnings=True,    # Suppress warnings
        stepwise=True,             # Use a stepwise approach
        information_criterion='aic' # Use AIC for model selection
    )

    # Display the best ARIMA parameters
    print("Best ARIMA model:", model.summary())

    # Fit the best ARIMA model to the data
    best_model = SARIMAX(price_data, order=model.order, enforce_stationarity=False, enforce_invertibility=False)
    best_model_fit = best_model.fit(disp=False)

    # Forecasting
    n_periods = 30  # Number of periods to forecast
    forecast = best_model_fit.get_forecast(steps=n_periods)
    forecast_ci = forecast.conf_int()

    # Plot the forecast
    plt.figure(figsize=(10, 5))
    plt.plot(price_data, label='Observed')
    plt.plot(forecast.predicted_mean, color='red', label='Forecast')
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.title("Brent Oil Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig('../figures/SARIMA__future_forcast.png', format='png', dpi=300)
    plt.show()

    print('End of SARIMA model')


def run_arima_model():
    # Loading the dataset
    data = pd.read_csv('../data/natural_gas/Brent_Oil_Prices.csv', parse_dates=['Date'], index_col='Date')

    print("ADF Test Results for Price:")
    adf_test(data['Price'])

    plot_differenced_price(data)
    arima_model(data)
    sarima_model(data)
