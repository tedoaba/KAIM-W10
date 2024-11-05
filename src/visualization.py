import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.dates as mdates
import pandas as pd

sns.set(style="whitegrid")

def plot_price_trend(data, save_path='../figures/price_trend.png'):
    """Plot the original price trend."""
    plt.figure(figsize=(20.5, 10))
    plt.plot(data.index, data['Price'], color='blue')
    plt.title('Brent Oil Price Trends')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(False)
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Figure saved as {save_path}")
    plt.show()

def plot_moving_averages(data, save_path='../figures/moving_averages.png'):
    """Plot the price trend with moving averages."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Price'], label='Original Data', color='blue')
    if 'SMA_3' in data:
        plt.plot(data.index, data['SMA_3'], label='3-Month Moving Average', color='orange')
    if 'SMA_6' in data:
        plt.plot(data.index, data['SMA_6'], label='6-Month Moving Average', color='green')
    if 'SMA_12' in data:
        plt.plot(data.index, data['SMA_12'], label='12-Month Moving Average', color='red')

    plt.title('Moving Averages (1990-2022)')
    plt.xlabel('Year')
    plt.ylabel('Import Value')
    plt.legend()
    plt.grid(False)

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Figure saved as {save_path}")

    plt.show()

def plot_prices(merged_data, save_path='../figures/merged_prices.png'):
    """Plot Natural Gas and Crude Oil Prices over time with two y-axes."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Natural Gas Prices
    ax1.plot(merged_data['Date'], merged_data['Gas_price'], color='coral', label='Natural gas')
    ax1.set_ylabel('Natural gas price', color='coral')
    ax1.tick_params(axis='y', labelcolor='coral')

    # Secondary y-axis for Crude Oil Prices, scaling by dividing by 9
    ax2 = ax1.twinx()
    ax2.plot(merged_data['Date'], merged_data['Oil_price'] / 9, color='deepskyblue', label='Crude oil')
    ax2.set_ylabel('Crude oil price', color='deepskyblue')
    ax2.tick_params(axis='y', labelcolor='deepskyblue')

    # Set x-axis label and limits
    ax1.set_xlabel('Date')
    ax1.set_xlim(merged_data['Date'].min(), merged_data['Date'].max())

    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Display the plot
    plt.title("Natural Gas and Crude Oil Prices Over Time")

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Figure saved as {save_path}")
    plt.show()

def plot_with_annotation(merged_data, save_path='../figures/annotations.png'):
    """Plot Natural Gas and Crude Oil Prices with annotation and highlight period."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Natural Gas Prices
    ax1.plot(merged_data['Date'], merged_data['Gas_price'], color='coral', label='Natural gas')
    ax1.set_ylabel('Natural gas price', color='coral')
    ax1.tick_params(axis='y', labelcolor='coral')

    # Secondary y-axis for Crude Oil Prices, scaling by dividing by 9
    ax2 = ax1.twinx()
    ax2.plot(merged_data['Date'], merged_data['Oil_price'] / 9, color='deepskyblue', label='Crude oil')
    ax2.set_ylabel('Crude oil price', color='deepskyblue')
    ax2.tick_params(axis='y', labelcolor='deepskyblue')

    # Add a red rectangle to highlight a period
    ax1.add_patch(
        patches.Rectangle(
            (mdates.date2num(pd.to_datetime('2001-03-01')), -100),
            mdates.date2num(pd.to_datetime('2001-11-01')) - mdates.date2num(pd.to_datetime('2001-03-01')),
            200,
            color='red',
            alpha=0.1
        )
    )

    # Annotate with a curved arrow and text
    ax1.annotate(
        'Early 2000s\nrecession',
        xy=(mdates.date2num(pd.to_datetime('2001-03-01')), 17),
        xytext=(mdates.date2num(pd.to_datetime('1999-01-01')), 15),
        arrowprops=dict(arrowstyle='->', color='black', lw=1, connectionstyle="arc3,rad=-0.5")
    )

    # Set x-axis label and title
    ax1.set_xlabel('Time')
    ax1.set_title('Natural gas price and US economic depression')

    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Figure saved as {save_path}")

    # Display the plot
    plt.show()
