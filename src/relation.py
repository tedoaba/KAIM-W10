import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


df =  '../data/new/Brent_Oil_Prices.csv'
exchange_rate_fred = '../data/new/usd_eur_exchange_rate_fred.csv'
exchange_rate_vintage = '../data/new/usd_eur_exchange_rates_alpha_vantage.csv'


def load_gdp(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(['Unnamed: 0'], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df
    

def relation_with_gdp(df, exchange_rate_fred, exchange_rate_vintage):
    # Load the datasets
    df = load_gdp(df)
    exchange_rate_fred = load_gdp(exchange_rate_fred)
    exchange_rate_vintage =  load_gdp(exchange_rate_vintage)

    # Merge datasets on 'Date' column using an outer join to keep all dates
    merged_data = df.merge(exchange_rate_fred, on='Date', how='outer') \
                    .merge(exchange_rate_vintage, on='Date', how='outer')

    # Sort by 'Date' to ensure data is in chronological order
    merged_data = merged_data.sort_values(by='Date')

    # Initialize a scaler
    scaler = MinMaxScaler()

    # Apply Min-Max Scaling to relevant columns, ignoring the 'Date' column
    columns_to_scale = ['Price', 'DEXUSEU', 'Open', 'High', 'Low', 'Close']
    merged_data[columns_to_scale] = scaler.fit_transform(merged_data[columns_to_scale])

    # Plotting
    plt.figure(figsize=(18, 8))

    # Plot each normalized column in a single plot with different colors and labels
    plt.plot(merged_data['Date'], merged_data['Price'], label='Brent Oil Price', color='blue')
    plt.plot(merged_data['Date'], merged_data['DEXUSEU'], label='USD/EUR Exchange Rate (FRED)', color='purple')
    plt.plot(merged_data['Date'], merged_data['Open'], label='USD/EUR Open (Alpha Vantage)', color='red')
    plt.plot(merged_data['Date'], merged_data['High'], label='USD/EUR High (Alpha Vantage)', color='pink')
    plt.plot(merged_data['Date'], merged_data['Low'], label='USD/EUR Low (Alpha Vantage)', color='brown')
    plt.plot(merged_data['Date'], merged_data['Close'], label='USD/EUR Close (Alpha Vantage)', color='gray')

    # Labels and title
    plt.xlabel('Date')
    plt.ylabel('Scaled Values')
    plt.title('Brent Oil Prices and USD/EUR Exchange Rates Over Time')
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.show()


relation_with_gdp(df, exchange_rate_fred, exchange_rate_vintage)