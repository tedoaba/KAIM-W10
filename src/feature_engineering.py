import pandas as pd

def add_time_features(data):
    """Add Month and Year features to the dataset based on index."""
    data['Month'] = data.index.month
    data['Year'] = data.index.year
    return data

def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split data into features (X) and target (y) and then into training and testing sets."""
    from sklearn.model_selection import train_test_split
    X = data[['Month', 'Year']]
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def generate_future_dates(last_date, periods=12):
    """Generate future dates for forecasting."""
    return pd.date_range(start=last_date, periods=periods + 1, freq='M')[1:]

def create_future_features(future_dates):
    """Create a DataFrame for future dates with Month and Year features."""
    return pd.DataFrame({'Month': future_dates.month, 'Year': future_dates.year})

def forecast_future(model, future_df):
    """Generate future predictions."""
    return model.predict(future_df)
