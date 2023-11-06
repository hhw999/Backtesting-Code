import pandas as pd
import yfinance as yf
import mplfinance as mpf
import bokeh
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import matplotlib.dates as mdates
from bokeh.plotting import figure, output_file
from bokeh.io import output_notebook, show
from bokeh.resources import INLINE
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, plot_heatmaps, resample_apply, barssince
from datetime import datetime, time, timedelta
from pytz import UTC  # Import the UTC timezone from pytz library
from tqdm import tqdm

def snap(target, bar):    
    if bar.name == pd.to_datetime(target) :
        print('Current bar', bar)
        print('Levels', Min15_high, Min15_low)
        print('fractal high, low', fractal_high, fractal_low)

def is_specific_candle_fractal(data, timestamp, HighLow):
    """Checks if the current bar is a fractal high/low.
        Input the price data as a pd dataframe, the timestamp of the bar in question and 1 or -1 to identify if you are looking to check if this is a fractal high or fractal low."""
    index = data.index.get_loc(timestamp)

    fractal_high = data['High'].iloc[index]
    fractal_low = data['Low'].iloc[index]

    start_index = max(0, index - 2)
    end_index = min(len(data) - 1, index + 2)

    is_high = fractal_high >= max(data['High'].iloc[start_index:end_index + 1])
    is_low = fractal_low <= min(data['Low'].iloc[start_index:end_index + 1])

    if HighLow == 1:
        if is_high:
            return 'High'
        else: return False
    
    elif HighLow == -1:
        if is_low:
            return 'Low'
        else: return False
    
    else:
        return False

def identify_bar(data):
    """Add a 'bar' column with 'Red' or 'Green' based on Close and Open prices."""
    data = data.copy()
    data['Bar'] = 'Green'  # Initialize 'bar' column with 'Green'
    red_mask = data['Close'] < data['Open']  # Create a mask for 'Red' bars
    data.loc[red_mask, 'Bar'] = 'Red'  # Set 'bar' to 'Red' for rows where Close < Open
    return data

def identify_fractals(data):
    """Identify fractals based off of code from Trading view"""
    
    # Fractal period
    fractal_period = 5
    fractal_dates = []

    def is_bill_williams_fractal(high, low, index):
        if index < 4 or index >= len(high) - 4:
            return False
        fractal_high = high[index]
        fractal_low = low[index]
        is_high = fractal_high >= max(high[index - 2:index + 3])
        is_low = fractal_low <= min(low[index - 2:index + 3]) 
        # is_high = fractal_high >= max(high[index - 2:index + 2]) and fractal_high >= max(high[index - 1:index + 4])
        # is_low = fractal_low <= min(low[index - 2:index + 2]) and fractal_low <= min(low[index - 1:index + 4])
        
        if is_high == True: return +1
        elif is_low == True: return -1

    # Return high or low fractal
    for i in range(len(data)):
        if is_bill_williams_fractal(data['High'], data['Low'], i) == +1:
            fractal_dates.append((data.index[i], 'fractal_high'))
        
        elif is_bill_williams_fractal(data['High'], data['Low'], i) == -1:
            fractal_dates.append((data.index[i], 'fractal_low'))

    data['Fractal'] = ''
    for date, fractal_type in fractal_dates:
        data.loc[date, 'Fractal'] = fractal_type

    data = data.dropna(subset=['Open'])
    data.index = data.index.tz_localize(None)

    return data

def set_15min_levels(current_bar):
    """Define critical levels needed for 15min entry."""
    
    global fractal_high, fractal_low, Min15_high, Min15_low, min15_record, fractal_high_validity, fractal_low_validity, recently_updated
    
    # set last 15 min fractal levels
    if current_bar['Fractal'] == "fractal_high":
        fractal_high = current_bar['High']
        fractal_high_validity = 1

        if is_specific_candle_fractal(m15_data_fractal, current_bar.name, -1) == 'Low':
            fractal_low = current_bar['Low'] 
            fractal_low_validity = 1


    if current_bar['Fractal'] == "fractal_low":
        fractal_low = current_bar['Low'] 
        fractal_low_validity = 1

        if is_specific_candle_fractal(m15_data_fractal, current_bar.name, 1) == 'High':
            fractal_low = current_bar['High'] 
            fractal_low_validity = 1

    # set 15 min range
    if current_bar['High'] >= Min15_high:
        Min15_high = current_bar['High']
        Min15_low = fractal_low 
        recently_updated = "up"

    elif current_bar['Low'] <=  Min15_low:
        Min15_low = current_bar['Low']
        Min15_high = fractal_high
        recently_updated = "down"

    else:
        recently_updated = None

    # snap('2023-09-14 11:45:00',current_bar)
    new_data = pd.DataFrame({'Datetime': [current_bar.name], 'M15_high': Min15_high, 'M15_low': Min15_low, 'Last_Fractal_High': fractal_high, 'Last_Fractal_Low': fractal_low})
    min15_record = pd.concat([min15_record, new_data], ignore_index=True) 
    
def check_time_range(current_time):
    """Function to check if the current_time is out of kill zone. Return True if in KZ."""
    # Convert the timestamp to UTC timezone
    # timestamp_utc = timestamp.tz_localize(UTC)

    # # Extract the time from the timestamp
    # current_time = timestamp_utc.time()

    # Define time ranges in UTC
    time_range_1_start = time(8, 0)
    time_range_1_end = time(10, 30)
    time_range_2_start = time(13, 0)
    time_range_2_end = time(15, 30)
    # Check if the time is within either of the defined ranges
    if (time_range_1_start <= current_time < time_range_1_end) or \
        (time_range_2_start <= current_time < time_range_2_end):
        return True
    else:
        return False

def check_sweep_origin(min15_data_fractal, current_index, search_color):
    """Checks where the liquidity sweep originates.
    Return True when the sweep origin is within the KZ."""
    last_opposite_color_index = None
    current_index_int = min15_data_fractal.index.get_loc(current_index)

    for index, row in min15_data_fractal.iloc[current_index_int-1::-1].iterrows():
        if row['Bar'] == search_color:
            last_opposite_color_index = min15_data_fractal.index[min15_data_fractal.index.get_loc(index)+1]
            break  # Exit the loop as soon as an opposite color is found

    # If no opposite color candle was found, return False
    if not last_opposite_color_index:
        return False

    # Convert the last opposite color change timestamp to UTC and extract its time
    last_opposite_color_time = last_opposite_color_index.tz_localize(UTC).time()
    
    # print('sweep check time:', last_opposite_color_time)

    return check_time_range(last_opposite_color_time)  # Returns true when in KZ

def check_15min_entry(data):
    """Check for an entry. Place Limit order if entry is found."""
    global m15_data_fractal, fractal_high, fractal_low, Min15_high, Min15_low, min15_record, fractal_high_validity, fractal_low_validity, recently_updated, Pending_orders

    if data['High'] > 0 and data['Low'] > 0 and Min15_high > 0 and Min15_low > 0. and fractal_high > 0 and fractal_low > 0:
       
        if data['High'] > Min15_high or (data['High'] > fractal_high and fractal_high_validity ==1):
            if check_time_range(data.name.time()) and check_sweep_origin(m15_data_fractal, data.name, "Red"):
            
                if recently_updated == "up":
                    recently_updated = False
                    set_15min_levels(data)
                    return

                SELL(data)

            fractal_high_validity = 0

        if data['Low'] < Min15_low or (data['Low'] < fractal_low and fractal_low_validity ==1):
            if check_time_range(data.name.time()) and check_sweep_origin(m15_data_fractal, data.name, "Green"):

                if recently_updated == "down":
                    recently_updated = False
                    set_15min_levels(data)
                    return

                BUY(data)
            
            fractal_low_validity = 0

    set_15min_levels(data)

def check_limit(data):
    global Pending_orders, Active_orders

    if not Pending_orders.empty:
        # Close Pending Orders
        if data.name.time() == time(10,30) or data.name.time() == time(15, 30):
            for index, order in Pending_orders.iterrows():
                Pending_orders.drop(index, inplace=True)  

        # Fill Stop Order if tagged in
        for index, order in Pending_orders.iterrows():
            if order['Type'] == 'BUY' and data['High'] > order['Limit']:

                new_order = pd.DataFrame({
                'Datetime': [order['Datetime']],
                'Type': [order['Type']], 
                'Limit': [order['Limit']], 
                'SL': [order['SL']],
                'TP': [order['TP']]})

                Active_orders = pd.concat([Active_orders, new_order], ignore_index=True)
                if index in Pending_orders.index:
                    Pending_orders.drop(index, inplace=True)
 

            elif order['Type'] == 'Sell' and data['Low'] < order['Limit']:
            

                new_order = pd.DataFrame({
                'Datetime': [order['Datetime']],
                'Type': [order['Type']], 
                'Limit': [order['Limit']], 
                'SL': [order['SL']],
                'TP': [order['TP']]})

                Active_orders = pd.concat([Active_orders, new_order], ignore_index=True)
                Pending_orders.drop(index, inplace=True)  

def check_close(data):
    global Active_orders, Closed_orders
    if not Active_orders.empty:

        for index, order in Active_orders.iterrows():
            if (order['Type'] == 'BUY' and data['High'] > order['TP']) or \
            (order['Type'] == 'SELL' and data['Low'] < order['TP']):
                
                new_order = pd.DataFrame({
                'Datetime': [order['Datetime']],
                'Type': [order['Type']], 
                'Limit': [order['Limit']], 
                'SL': [order['SL']],
                'TP': [order['TP']],
                'Close Datetime': [data.name],
                'Return': 3})

                Closed_orders = pd.concat([Closed_orders, new_order], ignore_index=True)
                Active_orders.drop(index, inplace=True)

            if (order['Type'] == 'BUY' and data['Low'] <= order['SL']) or \
            (order['Type'] == 'SELL' and data['High'] >= order['SL']):
                
                new_order = pd.DataFrame({
                'Datetime': [order['Datetime']],
                'Type': [order['Type']], 
                'Limit': [order['Limit']], 
                'SL': [order['SL']],
                'TP': [order['TP']],
                'Close Datetime': [data.name],
                'Return': -1})

                Closed_orders = pd.concat([Closed_orders, new_order], ignore_index=True)
                if index in Active_orders.index:
                    Active_orders.drop(index, inplace=True)

def BUY(data):
    global Pending_orders, Stop_Pips

    Long_StopLoss = round(data['Low'], 4) - Stop_Pips
    Long_TakeProfit = round((data['High'] + (3*(data['High']-Long_StopLoss))),4)

    # if Long_TakeProfit >= H4_high:
        # return

    # Add Limit Order to open_orders to track pending orders
    new_order = pd.DataFrame({
        'Datetime': [data.name],
        'Type': ['BUY'], 
        'Limit': [round(data['High'], 4)], 
        'SL': [Long_StopLoss],
        'TP': [Long_TakeProfit]})
    
    # print("new order", new_order)
    Pending_orders = pd.concat([Pending_orders, new_order], ignore_index=True)

def SELL(data):
    global Pending_orders, Stop_Pips

    Short_StopLoss = round(data['High'], 4) + Stop_Pips
    Short_TakeProfit = round((data['Low'] - (3*(Short_StopLoss-data['Low']))),4)

    # if Short_TakeProfit <= H4_low:
    #     return  

    # Add Limit Order to open_orders to track pending orders
    new_order = pd.DataFrame({
        'Datetime': [data.name],
        'Type': ['SELL'], 
        'Limit': [round(data['Low'], 4)], 
        'SL': [Short_StopLoss],
        'TP': [Short_TakeProfit]})
    
    # print("new order", new_order)
    Pending_orders = pd.concat([Pending_orders, new_order], ignore_index=True)
 
def process_results(Closed_orders):
    # Set 'Close Datetime' as the index
    Closed_orders.set_index('Close Datetime', inplace=True)

    # Group by month and sum the 'Return' column
    monthly_returns = Closed_orders['Return'].resample('M').sum().reset_index()

    print("Total Return:", Closed_orders['Return'].sum())
    print("Monthly Return Breakdown:\n", monthly_returns)

    # Plotting
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # 1. Line graph for cumulative sum of 'Return'
    Closed_orders['Return'].cumsum().plot(ax=ax[0])
    ax[0].set_title("Cumulative Sum of Returns")
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("Cumulative Return")

    # 2. Bar chart for monthly sum of 'Return'
    monthly_returns = Closed_orders['Return'].resample('M').sum()
    monthly_returns.plot(kind='bar', ax=ax[1])
    ax[1].set_title("Monthly Sum of Returns")
    ax[1].set_xlabel("Month")
    ax[1].set_ylabel("Return")
    plt.tight_layout()

    # plt.show()

print('GBPUSD Analysis')
df = pd.read_csv("/Users/hugowatkinson/Documents/Trading/Historical Data/gbpusd-m15-bid-2022-09-16-2023-09-16.csv")
# df = df.iloc[(-200):]
df['timestamp'] = pd.to_datetime(df["timestamp"], unit="ms") + pd.Timedelta(hours=1)
df.set_index('timestamp', inplace=True)
column_mapping = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
df = df.rename(columns=column_mapping)
df = df[df['Volume'] != 0]
m15_data_fractal_1 = identify_fractals(df)
m15_data_fractal = identify_bar(m15_data_fractal_1)

m15_data_fractal.to_csv('15Min_price.csv')

# Initialise variables
fractal_high, fractal_low, Min15_high, Min15_low, fractal_high_validity, fractal_low_validity = 0,0,0,0,0,0
min15_record = pd.DataFrame(columns=['Datetime','M15_high', 'M15_low'])
recently_updated = False
Stop_Pips = 0.0002
Pending_orders = pd.DataFrame(columns = ['Datetime', 'Type', 'Limit', 'SL', 'TP'])
Active_orders = pd.DataFrame(columns = ['Datetime', 'Type', 'Limit', 'SL', 'TP'])
Closed_orders = pd.DataFrame(columns = ['Datetime', 'Type', 'Limit', 'SL', 'TP', 'Close Datetime','Return'])

for  index, bar in tqdm(m15_data_fractal.iterrows()):
    check_close(bar)
    check_limit(bar)
    check_15min_entry(bar)

results = process_results(Closed_orders)

# print('Closed Orders:', Closed_orders.to_string())
min15_record['Datetime'] = pd.to_datetime(min15_record['Datetime'])
min15_record.set_index('Datetime', inplace=True)

# mpf.plot(df, type='candle', style='yahoo', title='Candlestick Chart', ylabel='Price')
# plt.show()

##############################################################################################################################

# min15_record['Datetime'] = pd.to_datetime(min15_record['Datetime'])
# min15_record.set_index('Datetime', inplace=True)

# m15_data_fractal.index = pd.to_datetime(m15_data_fractal.index)

# min15_record_w_Price = m15_data_fractal.merge(min15_record, how='left', left_index=True, right_index=True)



# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# # Plot on ax1 as you originally did:
# ax1.plot(min15_record_w_Price.index, min15_record_w_Price['M15_high'], label='M15 High', color='green')
# ax1.plot(min15_record_w_Price.index, min15_record_w_Price['M15_low'], label='M15 Low', color='red')
# ax1.set_ylabel('Price')
# ax1.set_ylim(1.23, 1.255)
# ax1.legend()

# # Plotting candlesticks on ax2 using matplotlib:

# # Define widths for bars
# bar_width = (min15_record_w_Price.index[1] - min15_record_w_Price.index[0]).total_seconds()/ (3 * 24 * 60 * 60)

# colors = min15_record_w_Price['Close'] >= min15_record_w_Price['Open']
# ax2.bar(min15_record_w_Price.index, min15_record_w_Price['High'] - min15_record_w_Price['Low'],
#         bottom=min15_record_w_Price['Low'], color='black', width=bar_width)
# ax2.bar(min15_record_w_Price.index, min15_record_w_Price['Close'] - min15_record_w_Price['Open'],
#         bottom=min15_record_w_Price[['Open', 'Close']].min(axis=1), color=colors.map({True: 'g', False: 'r'}),
#         width=bar_width)

# ax2.set_ylabel('Price')
# ax2.set_title('Candlestick Chart')

# # Rest of your formatting:
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))  # Format dates as you prefer
# ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
# plt.setp(ax1.get_xticklabels(), rotation=45)
# plt.setp(ax2.get_xticklabels(), rotation=45)
# ax1.grid(True)
# ax2.grid(True)
# plt.tight_layout()
# plt.show()

##############################################################################################################################
