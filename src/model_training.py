from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def initialize_xgboost():
    """Initialize an XGBoost Regressor model with specified parameters."""
    return XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0.01,
        reg_lambda=1,
        min_child_weight=1,
        booster='gbtree',
        random_state=42,
        verbosity=1
    )

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model with early stopping and evaluation metrics."""
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model using test data and return metrics as a dictionary."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}, y_pred


def display_metrics(model_name, metrics):
    """Display model evaluation metrics in a DataFrame."""
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'RMSE': [metrics['RMSE']],
        'MAE': [metrics['MAE']],
        'MSE': [metrics['MSE']],
        'R²': [metrics['R2']]
    })
    
    metrics_df.to_csv('../data/xgb_metrics.csv', index=True)
    print(metrics_df)
    return metrics_df

# Define the models to be tested
def get_models():
    return {
        'CatBoost': CatBoostRegressor(verbose=0, n_estimators=100, learning_rate=0.1),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(n_estimators=100),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    }

# Function to train models and make predictions
def train_and_predict(models, X_train, y_train, X_test):
    predictions = {}
    for model_name, model in models.items():
        print(f'Training {model_name}...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred
        print(f'{model_name} model trained and predictions made.')
    return predictions

# Function to forecast future values
def forecast_future_mul(models, data, X_train, y_train, future_dates):
    for model_name, model in models.items():
        print(f'Training {model_name} for forecasting...')
        model.fit(X_train, y_train)
        
        future_df = pd.DataFrame({'Month': future_dates.month, 'Year': future_dates.year})
        future_predictions = model.predict(future_df)

        plt.figure(figsize=(15, 6))
        plt.plot(data.index, data['Price'], label='Historical Data', color='blue')
        plt.plot(future_dates, future_predictions, label='Forecasted Data', color='orange', linestyle='--', marker='o')
        plt.title(f'Oil Prices Forecast with {model_name} (Historical + Future Predictions)')
        plt.xlabel('Year')
        plt.ylabel('Oil Price')
        plt.legend()
        plt.grid(False)
        plt.savefig(f'../figures/{model_name}_forcating_price.png', format='png', dpi=300)
        plt.show()

# Function to calculate evaluation metrics
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results[model_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2}

        results_csv = pd.DataFrame(results).T
        results_csv.to_csv('../data/evaluation_result.csv', index=True)
    return pd.DataFrame(results).T

