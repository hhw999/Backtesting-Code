# 4 Hour directional Bias Def Stats
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
        is_high = fractal_high >= max(high[index - 2:index + 2])
        is_low = fractal_low <= min(low[index - 2:index + 2]) 
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

def determine_directional_bias(data):
    global H4_high, H4_low, directional_bias, H4_high_from_fractal, H4_low_from_fractal, h4_data_fractal  # Declare these variables as global

    if H4_high > 0 and H4_low > 0:
        previous_h4_high = H4_high
        previous_h4_low = H4_low

        # Helper function to fetch the next available bar with data using relative indexing
        def get_next_bar_with_data(current_pos):
            next_pos = current_pos + 1  # Start with the very next position
            while next_pos < len(h4_data_fractal) and h4_data_fractal.iloc[next_pos].isna().all():  # Keep iterating if the row has all NaN values
                next_pos += 1
            if next_pos < len(h4_data_fractal):  # If we're still within the DataFrame bounds
                return h4_data_fractal.iloc[next_pos]
            else:
                print(f"No data available after position: {current_pos}")
                return None

        current_pos = h4_data_fractal.index.get_loc(data.name)
        
        next_h4_bar = get_next_bar_with_data(current_pos)  # Get the next available bar with data
        if next_h4_bar is not None:
            # If the next_h4_bar was found, try to find the bar after that
            next_pos_for_bar2 = h4_data_fractal.index.get_loc(next_h4_bar.name)
            next_h4_bar2 = get_next_bar_with_data(next_pos_for_bar2)  
        else:
            next_h4_bar2 = None

        if data['High'] >= previous_h4_high and H4_high_from_fractal: 
            ### Interaction with upper H4 Level
            
            if data['Close'] > previous_h4_high and next_h4_bar['Low'] > previous_h4_high: #3.4 next bar entirely above
                #Body close with bar2 body close above 
                directional_bias = 2  # Set bias as SFT to the up side)
            
            elif data['Close'] <= previous_h4_high:
                # Wick Close
                directional_bias = -1 # Set directional bias NFT to the up side
                if next_h4_bar['Close'] >= previous_h4_high or next_h4_bar2['Close'] >= previous_h4_high:
                    # bar2/bar3 body close
                    directional_bias = 1 # Set directional bias FT to the up side
            
            elif data['Close'] > previous_h4_high and next_h4_bar['Close'] <= previous_h4_high:
                # body close with bar2 closing below level
                directional_bias = -1 # Set directional bias NFT to the up side
                if next_h4_bar2['Close'] > previous_h4_high:
                    # bar3 closing above level after 1x body close
                    directional_bias = 1 # Set directional bias FT to the up side
            
            elif data['Close'] > previous_h4_high and next_h4_bar['Close'] > previous_h4_high:
                # body close with bar 2 closing above level
                directional_bias = 1 # set bias as FT to the up side

            else: print('error in setting directional bias up')
        
        elif data['Low'] <= previous_h4_low and H4_low_from_fractal: 
            ### Interaction with lower H4 Level

            if data['Close'] < previous_h4_low and next_h4_bar['Low'] < previous_h4_low : #3.4 next bar entirely above 
                # Body close with bar 2 body close below level
                directional_bias = -2 # Set bias a SFT to the down side
            
            elif data['Close'] >= previous_h4_low:
                # Wick close
                directional_bias = 1 # Set bias as NFT to the down side
                if next_h4_bar['Close'] <= previous_h4_low or next_h4_bar2['Close'] <= previous_h4_low:
                    # bar2/bar3 body close
                    directional_bias = -1 # Set bias as FT to the down side 

            elif data['Close'] < previous_h4_low and next_h4_bar['Close'] >= previous_h4_low:
                # body close with bar 2 closing above level
                directional_bias = 1 # Set bias as NFT to the down side
                if next_h4_bar2['Close'] < previous_h4_low:
                    #bar3 closing below level after 1x body close
                    directional_bias = -1 # Set bias as FT to the down side

            elif data['Close'] < previous_h4_low and next_h4_bar['Close'] < previous_h4_low : 
                # body close with bar 2 closing below level
                directional_bias = -1 # set bias as FT to the down side

            else: 
                print('error in setting directional bias down', data.name)
                print(data)
                print('preivous H4 Low', previous_h4_low)
                print('Next Bar', next_h4_bar)
                print('Next Bar2', next_h4_bar2)
                print(h4_data_fractal[data.name-4:data.name+4])
        
        elif data['High'] >= previous_h4_high and H4_high_from_fractal == False: pass 
        elif data['Low'] <= previous_h4_low and H4_low_from_fractal == False: pass

        else: print('Directional bias update error')

def screenshot(current_bar, breached=None, breached2=None):
    global H4_high, H4_low, directional_bias, H4_record, H4_last_fractal_high, H4_last_fractal_low

    # Create new row
    new_data = pd.DataFrame({
        'Datetime': [current_bar.name], 
        'H4_high': [H4_high], 
        'H4_low': [H4_low], 
        'Directional_bias': [directional_bias], 
        'Last_Fractal_High': [H4_last_fractal_high], 
        'Last_Fractal_Low': [H4_last_fractal_low], 
        'result': ['']  # initialize with empty string
    })

    # Check for breached variable and update previous rows if needed
    if breached is not None:
        for index, row in H4_record.iterrows():
            if row['H4_high'] == breached:
                if row['H4_low'] == breached2:
                    H4_record.at[index, 'result'] = 1

            elif row['H4_low'] == breached:
                if row['H4_high'] == breached2:

                    H4_record.at[index, 'result'] = -1

        
    # Append the new data to H4_record
    H4_record = pd.concat([H4_record, new_data], ignore_index=True)

def update_H4_range(current_bar):
    """Updates H4 range to define trading range"""
    global H4_high, H4_low, H4_record, directional_bias, H4_high_from_fractal, H4_low_from_fractal, h4_data  # Declare these variables as global

    if current_bar['High'] > H4_high:
        ### Breach of the 4H range to the up side
        determine_directional_bias(current_bar)
        # Log breach to the upside
        breached = H4_high
        breached2 = H4_low

        # Check if the current bar is a fractal high
        if current_bar['Fractal'] == 'fractal_high' or is_specific_candle_fractal(h4_data, current_bar.name, 1) == 'High':
            H4_high_from_fractal = True
        else: H4_high_from_fractal = False

        H4_high = current_bar['High']

        # Set the low after a breach to the fractal high
        if H4_low == 0: 
            H4_low = current_bar['Low']
            # Check if the current bar is a fractal low
            if current_bar['Fractal'] == 'fractal_low':
                H4_low_from_fractal = True
            else: H4_low_from_fractal = False 

        elif H4_last_fractal_low < current_bar['Low']:
            H4_low = H4_last_fractal_low
            H4_low_from_fractal = True
            if is_specific_candle_fractal(h4_data, current_bar.name, -1) == 'Low':
                H4_low= current_bar['Low']

        elif current_bar['Low'] < H4_last_fractal_low:
            H4_low = current_bar['Low']
            # Check if the current bar is a fractal low
            if current_bar['Fractal'] == 'fractal_low' or is_specific_candle_fractal(h4_data, current_bar.name, -1) == 'Low':
                H4_low_from_fractal = True
            else: H4_low_from_fractal = False 

        else:
            print('Breach to high with no low update')

        screenshot(current_bar, breached, breached2)
        breached = ''
        breached2 = ''

    if current_bar['Low'] < H4_low:
        ### Breach of the H4 range to the down side
        determine_directional_bias(current_bar)

        # Log breach to the down side
        breached = H4_low
        breached2 = H4_high

        # Check if the current bar is a fractal low
        if current_bar['Fractal'] == 'fractal_low' or is_specific_candle_fractal(h4_data, current_bar.name, -1) == 'Low':
            H4_low_from_fractal = True
        else: H4_low_from_fractal = False 

        H4_low = current_bar['Low']

        if H4_high == 0: 
            H4_high = current_bar['High']
            # Check if the current bar is a fractal high
            if current_bar['Fractal'] == 'fractal_high':
                H4_high_from_fractal = True
            else: H4_high_from_fractal = False

        elif H4_last_fractal_high > current_bar['High']:
            H4_high = H4_last_fractal_high
            H4_high_from_fractal = True
            if is_specific_candle_fractal(h4_data, current_bar.name, 1) == 'High' and current_bar['High']:
                H4_high = current_bar['High']
                H4_high_from_fractal = True

        elif current_bar['High'] > H4_last_fractal_high or is_specific_candle_fractal(h4_data, current_bar.name, 1) == 'High' and current_bar['High']:
            H4_high = current_bar['High']
            # Check if the current bar is a fractal high
            if current_bar['Fractal'] == 'fractal_high':
                H4_high_from_fractal = True
            else: H4_high_from_fractal = False

        else:
            print('Breach to low with no high update')

        screenshot(current_bar, breached, breached2)
        breached = ''
        breached2 = ''

def update_h4_fractals(data):
    """ Update the 'last fractals' """
    global H4_last_fractal_high, H4_last_fractal_low, h4_data   # Declare these variables as global
    # Initial update 4H last fractals
    if H4_last_fractal_high == 0 or H4_last_fractal_low == 0:
        H4_last_fractal_high = data['High']
        H4_last_fractal_low = data['Low']

    # Update latest 4H last fractals
    if data['Fractal'] == "fractal_high":
        H4_last_fractal_high = data['High']
        if is_specific_candle_fractal(h4_data, data.name, -1) == 'Low':
            H4_last_fractal_low == data['Low']
        screenshot(data)

    if data['Fractal'] == "fractal_low":
        H4_last_fractal_low = data['Low'] 
        if is_specific_candle_fractal(h4_data, data.name, 1) == 'High':
            H4_last_fractal_high == data['High']
        screenshot(data)
        
def plotH4(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(data['Datetime'], data['H4_high'], label='H4 High', color='green')
    ax1.plot(data['Datetime'], data['H4_low'], label='H4 Low', color='red')
    ax1.set_ylabel('Price')
    ax1.set_title('H4 High and Low Over Time')
    ax1.legend()
    ax2.plot(data['Datetime'], data['Directional_bias'], label='Directional Bias', color='blue')
    ax2.set_ylabel('Directional Bias')
    ax2.set_title('Directional Bias Over Time')
    ax2.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))  # Format dates as you prefer
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax1.get_xticklabels(), rotation=45)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    ax1.grid(True)
    ax2.grid(True)
    plt.tight_layout()
    mpf.plot(h4_data, type='candle', style='yahoo', title='Candlestick Chart', ylabel='Price')
    plt.show()

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

def process_results(df, daily_momentum_data):
    # Merge daily momentum data into df based on the index
    if isinstance(df.index, pd.DatetimeIndex):
        df['date'] = df.index.date
    else:
        df['date'] = pd.to_datetime(df['Datetime']).dt.date

    daily_momentum_data = daily_momentum_data.copy()
    daily_momentum_data.loc[:, 'date'] = daily_momentum_data.index.date

    # Merge daily momentum data into df based on the 'date'
    df = df.merge(daily_momentum_data, on='date', how='left')
    df.drop('date', axis=1, inplace=True)

    # Remove rows where Directional_bias is 0
    df = df[(df['Directional_bias'] != 0)]

    # Drop consecutive duplicates based on H4_high and H4_low
    df = df[df[['H4_high', 'H4_low']].ne(df[['H4_high', 'H4_low']].shift()).any(axis=1)]
    
    H4_record.to_csv('H4_levels_directional_bias_check.csv')
    df.to_csv('H4_Ranges_w_Results.csv')

    # Filter rows where Directional_bias and daily_momentum are both positive or both negative
    aligned_df = df[((df['Directional_bias'] > 0) & (df['daily_momentum'] > 0)) |
                    ((df['Directional_bias'] < 0) & (df['daily_momentum'] < 0))]
    
    # Count number of successes in the aligned_df
    positive_aligned_success = aligned_df[(aligned_df['Directional_bias'] > 0) & (aligned_df['result'] == 1)].shape[0]
    negative_aligned_success = aligned_df[(aligned_df['Directional_bias'] < 0) & (aligned_df['result'] == -1)].shape[0]

    high_prob_success = positive_aligned_success + negative_aligned_success
    
    # Calculate high probability success rate
    high_prob_rate = (high_prob_success / len(aligned_df)) * 100 if len(aligned_df) != 0 else 0

    # Exclude rows where Directional_bias and daily_momentum are aligned
    non_aligned_df = df[~((df['Directional_bias'] > 0) & (df['daily_momentum'] > 0)) &
                        ~((df['Directional_bias'] < 0) & (df['daily_momentum'] < 0))]

    # Calculate successful counts for Directional_bias from non_aligned_df
    positive_bias_success = non_aligned_df[(non_aligned_df['Directional_bias'] > 0) & (non_aligned_df['result'] == 1)].shape[0]
    negative_bias_success = non_aligned_df[(non_aligned_df['Directional_bias'] < 0) & (non_aligned_df['result'] == -1)].shape[0]
    
    total_bias_success = positive_bias_success + negative_bias_success
    Directional_bias_success_rate = (total_bias_success / len(non_aligned_df)) * 100 if len(non_aligned_df) != 0 else 0  # Handle potential division by zero
    
    # Calculate successful counts for daily_momentum from non_aligned_df
    positive_momentum_success = non_aligned_df[(non_aligned_df['daily_momentum'] > 0) & (non_aligned_df['result'] == 1)].shape[0]
    negative_momentum_success = non_aligned_df[(non_aligned_df['daily_momentum'] < 0) & (non_aligned_df['result'] == -1)].shape[0]
    
    total_momentum_success = positive_momentum_success + negative_momentum_success
    daily_momentum_success_rate = (total_momentum_success / len(non_aligned_df)) * 100 if len(non_aligned_df) != 0 else 0  # Handle potential division by zero
    
    return high_prob_rate, Directional_bias_success_rate, daily_momentum_success_rate

def update_daily_momentum(daily_data):
    # Initialize momentum and counter for breaches
    daily_momentum = 0  # 0 means no momentum, 1 means bullish, -1 means bearish

    # Create a new column in the dataframe for daily momentum
    daily_data['daily_momentum'] = 0  # Initialize all to 0

    # Keep track of the last breached fractal high and low
    last_fractal_high = None
    last_fractal_low = None
    Unconfirmed_low = None 
    Unconfirmed_high = None
    previous_fractal_high = None
    previous_fractal_low = None

    # Loop through the dataframe
    for i, bar in daily_data.iterrows():

        if last_fractal_high ==  None and bar['Fractal'] == 'fractal_high': 
            last_fractal_high = bar['High']

        if last_fractal_low == None and bar['Fractal'] == 'fractal_low': 
            last_fractal_low = bar['Low']
        
        if last_fractal_low and last_fractal_high is not None:
            
            if Unconfirmed_low is not None:
                # Second Low Breach
                if bar['Low'] < Unconfirmed_low:
                    daily_momentum = -1
                    Unconfirmed_low = None

            if bar['Fractal'] == 'fractal_low':
                if bar['Low'] < last_fractal_low:
                    if daily_momentum == 1 or daily_momentum == 0:
                        if Unconfirmed_low == None:
                            # First Low Breach
                            Unconfirmed_low = bar['Low']
                        
                        if Unconfirmed_high is not None:
                            Unconfirmed_high = None

                previous_fractal_low = bar['Low']


            if bar['Low'] < last_fractal_low:
                if daily_momentum == -1:
                    last_fractal_low = bar['Low']
                    last_fractal_high = previous_fractal_high     

        
            if bar['Fractal'] == 'fractal_high':
                if bar['High'] > last_fractal_high:
                    if daily_momentum == -1 or daily_momentum == 0:
                        if Unconfirmed_high == None:
                            # First Low Breach
                            Unconfirmed_high = bar['High']
                        
                        if Unconfirmed_low is not None:
                            Unconfirmed_low = None

                    if daily_momentum == 1:
                        last_fractal_high = bar['High']
                        last_fractal_low = previous_fractal_low 
                
                previous_fractal_high = bar["High"]

            if bar['High'] > last_fractal_high:
                if Unconfirmed_high is not None:
                    # Second Low Breach
                    if bar['High'] > Unconfirmed_high:
                        daily_momentum = 1


        if daily_momentum != 0:
            daily_data.at[i, 'daily_momentum'] = daily_momentum

    return daily_data[['daily_momentum']]  # Return only 'daily_momentum' column


#####################################################################################################################

# Import 15 min data
# df = pd.read_csv("/Users/hugowatkinson/Documents/Trading/Historical Data/gbpusd-m15-bid-2022-09-16-2023-09-16.csv")
# df = df.iloc[(-20*100):]

print('EURUSD analysis')
df = pd.read_csv("/Users/hugowatkinson/Documents/Trading/Historical Data/eurusd-m15-bid-2020-09-16-2023-09-16.csv")
df['timestamp'] = pd.to_datetime(df["timestamp"], unit = "ms")
df.set_index('timestamp', inplace=True)
column_mapping = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
df = df.rename(columns=column_mapping)
# Remove rows where 'Volume' is equal to 0
df = df[df['Volume'] != 0]

# Resample to 4-hour data with custom time offset
h4_data = df.resample('4H', offset='2H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Resample to daily data
d1_data = df.resample('1D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Identify fractal H/L for all data (A bit of look ahead bias here?)
h4_data_fractal_1 = identify_fractals(h4_data)
h4_data_fractal = identify_bar(h4_data_fractal_1)
d1_data_fractal_1 = identify_fractals(d1_data)
d1_data_fractal = identify_bar(d1_data_fractal_1)

# Identify daily momentum
d1_data_momentum = update_daily_momentum(d1_data_fractal)

# Create the H4 record 
H4_record = pd.DataFrame(columns=['Datetime','H4_high', 'H4_low','Last_Fractal_High','Last_Fractal_Low', 'Directional_bias'])

# H4 Variables
H4_last_fractal_high = 0
H4_last_fractal_low = 0
H4_high = 0
H4_low = 0
H4_high_from_fractal = False
H4_low_from_fractal = False
H4_equ = 0
directional_bias = 0

#### Main Loop ###
for index, bar in h4_data_fractal.iterrows():
    update_H4_range(bar)
    update_h4_fractals(bar)

high_prob_rate, directional_bias_success_rate, daily_momentum_success_rate = process_results(H4_record, d1_data_momentum)
print(f"High Probability Success Rate: {high_prob_rate:.2f}%")
print(f"4H Directional Bias Success Rate: {directional_bias_success_rate:.2f}%")
print(f"Daily Momentum Success Rate: {daily_momentum_success_rate:.2f}%")


print('GBPUSD analysis')
df = pd.read_csv("/Users/hugowatkinson/Documents/Trading/Historical Data/gbpusd-m15-bid-2020-09-16-2023-09-16.csv")
df['timestamp'] = pd.to_datetime(df["timestamp"], unit = "ms")
df.set_index('timestamp', inplace=True)
column_mapping = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
df = df.rename(columns=column_mapping)
# Remove rows where 'Volume' is equal to 0
df = df[df['Volume'] != 0]

# Resample to 4-hour data with custom time offset
h4_data = df.resample('4H', offset='2H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Resample to daily data
d1_data = df.resample('1D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Identify fractal H/L for all data (A bit of look ahead bias here?)
h4_data_fractal_1 = identify_fractals(h4_data)
h4_data_fractal = identify_bar(h4_data_fractal_1)
d1_data_fractal_1 = identify_fractals(d1_data)
d1_data_fractal = identify_bar(d1_data_fractal_1)

# Identify daily momentum
d1_data_momentum = update_daily_momentum(d1_data_fractal)

# Create the H4 record 
H4_record = pd.DataFrame(columns=['Datetime','H4_high', 'H4_low','Last_Fractal_High','Last_Fractal_Low', 'Directional_bias'])

# H4 Variables
H4_last_fractal_high = 0
H4_last_fractal_low = 0
H4_high = 0
H4_low = 0
H4_high_from_fractal = False
H4_low_from_fractal = False
H4_equ = 0
directional_bias = 0

#### Main Loop ###
for index, bar in h4_data_fractal.iterrows():
    update_H4_range(bar)
    update_h4_fractals(bar)

high_prob_rate, directional_bias_success_rate, daily_momentum_success_rate = process_results(H4_record, d1_data_momentum)
print(f"High Probability Success Rate: {high_prob_rate:.2f}%")
print(f"4H Directional Bias Success Rate: {directional_bias_success_rate:.2f}%")
print(f"Daily Momentum Success Rate: {daily_momentum_success_rate:.2f}%")



####################################################################################################

# plotH4(H4_record)

# mpf.plot(h4_data, type='candle', style='yahoo', title='Candlestick Chart', ylabel='Price')
# plt.show()

# print('H4 Record', H4_record.to_string())

            # if data.name == datetime(2023, 8, 30, 6, 0, 0):
            #     print('current bar:', data)
            #     print('H4 High:', H4_high)
            #     print('H4_low:', H4_low)
            #     print('Next H4 Bar:', next_h4_bar)