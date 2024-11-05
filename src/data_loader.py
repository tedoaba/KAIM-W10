import pandas as pd

def load_data(filepath):
    """Load the CSV data and convert the Date column to datetime."""
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
    data.dropna(subset=['Date'], inplace=True)
    data.set_index('Date', inplace=True)
    return data

def load_datasets(filepath):
    """Load CSV data and convert the Date column to datetime."""
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
    return data

def rename_columns(data, column_map):
    """Rename columns in the data."""
    return data.rename(columns=column_map)