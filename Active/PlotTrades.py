import pandas as pd
import plotly.graph_objects as go
import os
import shutil

# Function to create candlestick plot for each trade
def create_candlestick_plot(trade, price_data, output_folder, num_bars_before):
    trade_start1 = trade['Sweep_DateTime']
    trade_start = trade_start1 - pd.DateOffset(minutes=num_bars_before * 15)

    trade_end = trade['Close_DateTime']

    # Filter price data for the given trade period
    trade_price_data = price_data.loc[trade_start:trade_end]

    # Create candlestick plot
    fig = go.Figure(data=[go.Candlestick(x=trade_price_data.index,
                                         open=trade_price_data['Open'],
                                         high=trade_price_data['High'],
                                         low=trade_price_data['Low'],
                                         close=trade_price_data['Close'])])

    fig.update_layout(title=f'Candlestick Plot - {trade_start} to {trade_end}',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      template='plotly_dark')

    # Save the plot to the output folder
    plot_filename = f"{output_folder}/candlestick_plot_{trade_start}_{trade_end}.html"
    fig.write_html(plot_filename)

# Function to process trades and create plots
def process_trades(order_list_file, price_data_file, output_folder, num_bars_before):
        
    # Clear the contents of the output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)    
   
    # Load order list
    order_list = pd.read_csv(order_list_file, parse_dates=['Sweep_DateTime', 'Close_DateTime'])

    # Load price data
    price_data = pd.read_csv(price_data_file)

    price_data['timestamp'] = pd.to_datetime(price_data["timestamp"], unit = "ms")
    price_data.set_index('timestamp', inplace=True)
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    price_data = price_data.rename(columns=column_mapping)
    # Remove rows where 'Volume' is equal to 0
    price_data = price_data[price_data['Volume'] != 0]

    # Process each trade and create a plot
    for index, trade in order_list.iterrows():
        create_candlestick_plot(trade, price_data, output_folder, num_bars_before)

    print("Charts Ready!")


if __name__ == "__main__":
    pair = 'eurusd'
    order_list_file = '/Users/hugowatkinson/Documents/Trading/Backtesting Code/Active/Output/Order_List.csv'
    price_data_file = f'/Users/hugowatkinson/Documents/Trading/Historical Data/{pair}-m15-bid-2020-09-16-2023-09-16.csv'
    output_folder = '/Users/hugowatkinson/Documents/Trading/Backtesting Code/Active/Output/output_plots'


    process_trades(order_list_file, price_data_file, output_folder, 30)