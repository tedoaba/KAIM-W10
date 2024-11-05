import pandas as pd

def calculate_moving_averages(data, windows=[3, 6, 12]):
    """Calculate Simple Moving Averages for given window sizes."""
    for window in windows:
        data[f'SMA_{window}'] = data['Price'].rolling(window=window).mean()
    return data

def get_date_range(data):
    """Get the minimum and maximum dates in the Date column."""
    return data['Date'].min(), data['Date'].max()

def filter_by_date_range(data, date_min, date_max):
    """Filter data by a specified date range."""
    return data[(data['Date'] >= date_min) & (data['Date'] <= date_max)]

def merge_datasets(oil_data, gas_data):
    """Merge oil and gas datasets on the Date column."""
    date_min, date_max = get_date_range(gas_data)
    filtered_oil_data = filter_by_date_range(oil_data, date_min, date_max)
    merged_data = pd.merge(filtered_oil_data, gas_data, on='Date', how='left')
    return merged_data
