from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    print(metrics_df)
    return metrics_df
