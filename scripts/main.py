import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('../src'))

from data_loader import load_data, load_datasets, rename_columns
from analysis import calculate_moving_averages, merge_datasets
from visualization import plot_price_trend, plot_moving_averages, plot_prices, plot_with_annotation



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



if __name__ == "__main__":
    oil_path = '../data/Brent_Oil_Prices.csv'
    gas_path = '../data/natural_gas_daily.csv'
    main(oil_path, gas_path)
