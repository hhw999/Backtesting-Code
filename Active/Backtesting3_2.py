# 4 Hour directional Bias Def Stats
import pandas as pd
from pandas import Timedelta
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import matplotlib.dates as mdates
from datetime import datetime, time, timedelta
from pytz import UTC  # Import the UTC timezone from pytz library
from tqdm import tqdm
import os
import shutil
import ProcessAll
import warnings

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
        fractal_high = high.iloc[index]
        fractal_low = low.iloc[index]
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

def determine_directional_bias(data):
    global H4_high, H4_low, directional_bias, H4_high_from_fractal, H4_low_from_fractal, h4_data_fractal  # Declare these variables as global

    if H4_high > 0 and H4_low > 0:
        previous_h4_high = H4_high
        previous_h4_low = H4_low

        # print('current bar has no bias assigned?', h4_data_fractal.loc[data.name, 'Directional_bias'] is None)

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

        if data['High'] >= previous_h4_high and H4_high_from_fractal and h4_data_fractal.loc[data.name, 'Directional_bias'] == None: 
            ### Interaction with upper H4 Level
            
            if ((next_h4_bar is not None) and (next_h4_bar2 is not None)) and (isinstance(next_h4_bar, pd.DataFrame) or isinstance(next_h4_bar, pd.Series)):
                if all(col in next_h4_bar.index for col in ['High', 'Low', 'Open', 'Close']):

                    if data['Close'] > previous_h4_high and next_h4_bar['High'] > previous_h4_high and next_h4_bar['Close'] > data['High']:
                        #Body close with bar2 body close above 
                        h4_data_fractal.loc[data.name, 'Directional_bias'] = 0
                        h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias'] = 2  # Set bias as SFT to the up side)
                    
                    elif data['Close'] <= previous_h4_high: #3.9 CHANGE <= to <
                        # Wick Close
                        h4_data_fractal.loc[data.name, 'Directional_bias'] = -1 # Set directional bias NFT to the up side
                        if next_h4_bar['Close'] >= previous_h4_high:
                            # bar2/bar3 body close
                            h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias'] = 1 # Set directional bias FT to the up side
                        elif next_h4_bar2['Close'] >= previous_h4_high:
                            # bar2/bar3 body close
                            h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias'] = -1
                            h4_data_fractal.loc[next_h4_bar2.name, 'Directional_bias'] = 1
                    
                    elif data['Close'] > previous_h4_high and next_h4_bar['Close'] <= previous_h4_high:
                        # body close with bar2 closing below level
                        h4_data_fractal.loc[data.name, 'Directional_bias'] = 0
                        h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias'] = -1 # Set directional bias NFT to the up side
                        if next_h4_bar2['Close'] > previous_h4_high:
                            # bar3 closing above level after 1x body close
                            h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias'] = 1 # Set directional bias FT to the up side
                    
                    elif data['Close'] > previous_h4_high and next_h4_bar['Close'] > previous_h4_high:
                        # body close with bar 2 closing above level
                        h4_data_fractal.loc[data.name, 'Directional_bias'] = 0
                        h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias'] = 1 # set bias as FT to the up side

                    else: print('error in setting directional bias up')
                else: print('Error next_h4_bar is not a valid dataframe with correct columns', next_h4_bar)
            else: print('Error: next_h4_bar is not a valid DataFrame', next_h4_bar)
        
        elif data['Low'] <= previous_h4_low and H4_low_from_fractal and h4_data_fractal.loc[data.name, 'Directional_bias'] == None: 
            ### Interaction with lower H4 Level

            if ((next_h4_bar is not None) and (next_h4_bar2 is not None)) and (isinstance(next_h4_bar, pd.DataFrame) or isinstance(next_h4_bar, pd.Series)):
                if all(col in next_h4_bar.index for col in ['High', 'Low', 'Open', 'Close']):
                
                    if data['Close'] < previous_h4_low and next_h4_bar['Low'] < previous_h4_low and next_h4_bar['Close'] < data['Low']: #3.11 CHANGE sft, next_h4_bar fully below level
                        # Body close with bar 2 body close below level
                        h4_data_fractal.loc[data.name, 'Directional_bias'] = 0
                        h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias']= -2 # Set bias a SFT to the down side
                    
                    elif data['Close'] >= previous_h4_low:
                        # Wick close
                        h4_data_fractal.loc[data.name, 'Directional_bias'] = 1 # Set bias as NFT to the down side
                        if next_h4_bar['Close'] <= previous_h4_low:
                            # bar2/bar3 body close
                            h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias'] = -1 # Set bias as FT to the down side 
                        elif next_h4_bar2['Close'] <= previous_h4_low:
                            # bar2/bar3 body close
                            h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias'] = 1
                            h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias'] = -1

                    elif data['Close'] < previous_h4_low and next_h4_bar['Close'] >= previous_h4_low:
                        # body close with bar 2 closing above level
                        h4_data_fractal.loc[data.name, 'Directional_bias'] = 0
                        h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias'] = 1 # Set bias as NFT to the down side
                        if next_h4_bar2['Close'] < previous_h4_low:
                            #bar3 closing below level after 1x body close
                            h4_data_fractal.loc[next_h4_bar2.name, 'Directional_bias'] = -1 # Set bias as FT to the down side

                    elif data['Close'] < previous_h4_low and next_h4_bar['Close'] < previous_h4_low:
                        # body close with bar 2 closing below level
                        h4_data_fractal.loc[data.name, 'Directional_bias'] = 0
                        h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias'] = -1 # set bias as FT to the down side

                    else: print('error in setting directional bias down')
                else: print('Error next_h4_bar is not a valid dataframe with correct columns', next_h4_bar)
            else: print('Error: next_h4_bar is not a valid DataFrame', next_h4_bar)
        
        elif data['High'] >= previous_h4_high and H4_high_from_fractal == False and (h4_data_fractal.loc[data.name, 'Directional_bias'] == None): h4_data_fractal.loc[data.name, 'Directional_bias'] = directional_bias
        elif data['Low'] <= previous_h4_low and H4_low_from_fractal == False and (h4_data_fractal.loc[data.name, 'Directional_bias'] == None): h4_data_fractal.loc[data.name, 'Directional_bias'] = directional_bias
        elif h4_data_fractal.loc[data.name, 'Directional_bias'] == 2 or -2:
            pass
        
        else: print('Directional bias update error', data.name)
    
        # print('H4 record directional bias on close', h4_data_fractal.loc[data.name, 'Directional_bias'])
        # print('H4 record directional bias on close', h4_data_fractal.loc[next_h4_bar.name, 'Directional_bias']) 
        # print('H4 record directional bias on close', h4_data_fractal.loc[next_h4_bar2.name, 'Directional_bias'])

def update_H4_range(current_bar):
    """Updates H4 range to define trading range"""
    global H4_high, H4_low, H4_record, directional_bias, H4_high_from_fractal, H4_low_from_fractal, h4_data  # Declare these variables as global

    if current_bar['High'] > H4_high:
        ### Breach of the 4H range to the up side
        determine_directional_bias(current_bar)
        directional_bias = h4_data_fractal.loc[current_bar.name, 'Directional_bias']

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

        elif current_bar['Low'] <= H4_last_fractal_low:
            H4_low = current_bar['Low']
            # Check if the current bar is a fractal low
            if current_bar['Fractal'] == 'fractal_low' or (is_specific_candle_fractal(h4_data, current_bar.name, -1) == 'Low' and current_bar['Low']):
                H4_low_from_fractal = True
            else: H4_low_from_fractal = False 

        else:
            print('Breach to high with no low update', current_bar['Low'], H4_last_fractal_low)

        # screenshot(current_bar, breached, breached2)
        breached = ''
        breached2 = ''

    if current_bar['Low'] < H4_low:
        ### Breach of the H4 range to the down side
        determine_directional_bias(current_bar)
        directional_bias = h4_data_fractal.loc[current_bar.name, 'Directional_bias']


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

        elif current_bar['High'] >= H4_last_fractal_high or is_specific_candle_fractal(h4_data, current_bar.name, 1) == 'High' and current_bar['High']:
            H4_high = current_bar['High']
            # Check if the current bar is a fractal high
            if current_bar['Fractal'] == 'fractal_high':
                H4_high_from_fractal = True
            else: H4_high_from_fractal = False

        else:
            print('Breach to low with no high update', current_bar['High'], H4_last_fractal_high)

        # screenshot(current_bar, breached, breached2)
        breached = ''
        breached2 = ''
    
    else: h4_data_fractal.loc[current_bar.name, 'Directional_bias'] = directional_bias
        
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

    if data['Fractal'] == "fractal_low":
        H4_last_fractal_low = data['Low'] 
        if is_specific_candle_fractal(h4_data, data.name, 1) == 'High':
            H4_last_fractal_high == data['High']

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
    with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            min15_record = pd.concat([min15_record, new_data], ignore_index=True)

def check_time_range(current_time):
    """Function to check if the current_time is out of kill zone. Return True if in KZ."""

    # Define time ranges in UTC
    time_range_1_start = time(0, 0)
    time_range_1_end = time(12, 0)
    time_range_2_start = time(12, 0)
    time_range_2_end = time(23, 0)
    # Check if the time is within either of the defined ranges
    if (time_range_1_start <= current_time < time_range_1_end) or \
        (time_range_2_start <= current_time < time_range_2_end):
        return True
    else:
        return False

    # # Define time ranges in UTC
    # time_range_1_start = time(0, 0)
    # time_range_1_end = time(12, 0)
    # time_range_2_start = time(12, 0)
    # time_range_2_end = time(23, 0)
    # # Check if the time is within either of the defined ranges
    # if (time_range_1_start <= current_time < time_range_1_end) or \
    #     (time_range_2_start <= current_time < time_range_2_end):
    #     return True
    # else:
    #     return False

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

def get_last_daily_momentum(before_datetime):
    # Filter out rows after the current 15-min datetime
    filtered_df = d1_data_momentum[d1_data_momentum.index < before_datetime]

    # Return the daily_momentum of the last row, if the filtered_df is not empty
    if not filtered_df.empty:
        return filtered_df.iloc[-1]['daily_momentum']
    return None

def check_15min_entry(data, Stop_Pips, m15_res):
    """Check for an entry. Place Limit order if entry is found."""
    global Momentum, m15_data_fractal, fractal_high, fractal_low, Min15_high, Min15_low, min15_record, fractal_high_validity, fractal_low_validity, recently_updated, Pending_orders, H4_high, H4_low, directional_bias, H4_high_from_fractal, H4_low_from_fractal, h4_data_fractal, d1_data_momentum  # Declare these variables as global
    
    Momentum = get_last_daily_momentum(data.name)
    dead_high = H4_low + ((H4_high - H4_low) * 0.55)
    dead_low = H4_low + ((H4_high - H4_low) * 0.45)

    # Continued entries
    if not Pending_orders.empty and check_time_range(data.name.time()):
        for index, order in Pending_orders.iterrows():
            if (pd.to_datetime(order['Sweep_DateTime']) == data.name - Timedelta(minutes=(m15_res))) and \
                                                        (((order['Buy/Sell'] == 'BUY') and (data['Low'] < (order['SL'] + Stop_Pips)) and (data['High'] < order['Limit']))):

                Pending_orders.drop(index, inplace=True)
                BUY(data, Stop_Pips,(order['SL'] + Stop_Pips), "Continued")

            if (pd.to_datetime(order['Sweep_DateTime']) == data.name - Timedelta(minutes=(m15_res))) and \
                                                        (((order['Buy/Sell'] == 'SELL') and (data['High'] > (order['SL']- Stop_Pips)) and (data['Low'] > order['Limit']))):

                Pending_orders.drop(index, inplace=True)
                SELL(data, Stop_Pips, (order['SL']- Stop_Pips), "Continued")
    
    if not Potential_orders.empty and check_time_range(data.name.time()):
        for index, order in Potential_orders.iterrows():
            if (pd.to_datetime(order['Sweep_DateTime']) == data.name - Timedelta(minutes=(m15_res))) and \
                                                        (((order['Buy/Sell'] == 'BUY') and (data['Low'] < (order['SL'] + Stop_Pips)) and (data['High'] < order['Limit']))) \
                                                            and (data['High'] < dead_low) \
                                                                and (Momentum != 0):
                # can add a "data['Bar'] == "Green" for order["type"] == "SELL" to ensure close above open???? 

                if index in Potential_orders.index:
                    Potential_orders.drop(index, inplace=True)
                else: 
                    for index, order in Potential_orders.iterrows():
                        Potential_orders.drop(index, inplace=True)                
                
                BUY(data, Stop_Pips, order['Liquidity_Level'], "Potential")

            elif (pd.to_datetime(order['Sweep_DateTime']) == data.name - Timedelta(minutes=(m15_res))) and \
                                                        (((order['Buy/Sell'] == 'SELL') and (data['High'] > (order['SL']- Stop_Pips)) and (data['Low'] > order['Limit']))) \
                                                            and (data['Low'] > dead_high) \
                                                                and (Momentum != 0):
                # can add a "data['Bar'] == "Green" for order["type"] == "SELL" to ensure close above open???? 

                if index in Potential_orders.index:
                    Potential_orders.drop(index, inplace=True)
                else: 
                    for index, order in Potential_orders.iterrows():
                        Potential_orders.drop(index, inplace=True)
                
                SELL(data, Stop_Pips, order['Liquidity_Level'], "Potential")

            elif (pd.to_datetime(order['Sweep_DateTime']) == data.name - Timedelta(minutes=(m15_res))) and \
                                                        (((order['Buy/Sell'] == 'BUY') and (data['Low'] < (order['SL'] + Stop_Pips)) and (data['High'] < order['Limit']))) \
                                                            and (data['High'] >= dead_low):
                # can add a "data['Bar'] == "Green" for order["type"] == "SELL" to ensure close above open???? 

                if index in Potential_orders.index:
                    Potential_orders.drop(index, inplace=True)
                else: 
                    for index, order in Potential_orders.iterrows():
                        Potential_orders.drop(index, inplace=True)
                
                Potential_BUY(data, Stop_Pips, order['Liquidity_Level'], "Potential")

            elif (pd.to_datetime(order['Sweep_DateTime']) == data.name - Timedelta(minutes=(m15_res))) and \
                                                        (((order['Buy/Sell'] == 'SELL') and (data['High'] > (order['SL']- Stop_Pips)) and (data['Low'] > order['Limit']))) \
                                                            and (data['Low'] <= dead_high):
                # can add a "data['Bar'] == "Green" for order["type"] == "SELL" to ensure close above open???? 

                if index in Potential_orders.index:
                    Potential_orders.drop(index, inplace=True)
                else: 
                    for index, order in Potential_orders.iterrows():
                        Potential_orders.drop(index, inplace=True)
                
                Potential_SELL(data, Stop_Pips, order['Liquidity_Level'], "Potential")

            elif (order['Buy/Sell'] == 'BUY' and data['Low'] < order['SL']):
                Potential_orders.drop(index, inplace=True)

            elif (order['Buy/Sell'] == 'SELL' and data['High'] > order['SL']):
                Potential_orders.drop(index, inplace=True)
            
    # Main entries
    if data['High'] > 0 and data['Low'] > 0 and \
        data['High'] < H4_high and data['Low'] > H4_low and \
        Min15_high > 0 and Min15_low > 0 and \
        fractal_high > 0 and fractal_low > 0:
        # Not sure if needed, but I guess this stops entries if the current bar is out of H4 range. 
        # Could need a way to stop all entries until H4 is updated?

        # m15 level sweep
        if data['High'] > Min15_high:
            if check_time_range(data.name.time()) and check_sweep_origin(m15_data_fractal, data.name, "Red") and (Momentum != 0):
                
                if recently_updated == "up":
                    recently_updated = False
                    set_15min_levels(data)
                    return
                
                if ((directional_bias < 0 or Momentum == -1) and (data['Low'] > dead_high)) or (directional_bias == -2):
                    # if directional_bias == 2: return
                    SELL(data, Stop_Pips, Min15_high, "M15Sweep")

                elif ((directional_bias < 0 or Momentum == -1) and (data['Low'] <= dead_high)):
                    # if directional_bias == 2: return
                    Potential_SELL(data, Stop_Pips, Min15_high, "Potential")

            fractal_high_validity = 0

        if data['Low'] < Min15_low:
            if check_time_range(data.name.time()) and check_sweep_origin(m15_data_fractal, data.name, "Green") and (Momentum != 0):
                
                if recently_updated == "down":
                    recently_updated = False
                    set_15min_levels(data)
                    return
                
                if ((directional_bias > 0 or Momentum == 1) and (data['High'] < dead_low)) or (directional_bias == 2):
                    # if directional_bias == -2: return
                    BUY(data, Stop_Pips, Min15_low, "M15Sweep")

                elif ((directional_bias > 0 or Momentum == 1) and (data['High'] >= dead_low)):
                    # if directional_bias == -2: return
                    Potential_BUY(data, Stop_Pips, Min15_low, "Potential")
            
            fractal_low_validity = 0

        # Fractal Sweep
        if (data['High'] > fractal_high and fractal_high_validity ==1):
            if check_time_range(data.name.time()) and check_sweep_origin(m15_data_fractal, data.name, "Red") and (Momentum != 0):
                
                if recently_updated == "up":
                    recently_updated = False
                    set_15min_levels(data)
                    return
            
                if ((directional_bias < 0 or Momentum == -1) and (data['Low'] > dead_high)) or (directional_bias == -2):
                    # if directional_bias == 2: return
                    SELL(data, Stop_Pips, fractal_high, "FractalSweep")

                elif ((directional_bias < 0 or Momentum == -1) and (data['Low'] <= dead_high)):
                    # if directional_bias == 2: return
                    Potential_SELL(data, Stop_Pips, fractal_high, "Potential")

            fractal_high_validity = 0

        if (data['Low'] < fractal_low and fractal_low_validity ==1):
            if check_time_range(data.name.time()) and check_sweep_origin(m15_data_fractal, data.name, "Green") and (Momentum != 0):
                
                if recently_updated == "down":
                    recently_updated = False
                    set_15min_levels(data)
                    return

                if ((directional_bias > 0 or Momentum == 1) and (data['High'] < dead_low)) or (directional_bias == 2):
                    # if directional_bias == -2: return
                    BUY(data, Stop_Pips, fractal_low, "FractalSweep")

                elif ((directional_bias > 0 or Momentum == 1) and (data['High'] >= dead_low)):
                    # if directional_bias == -2: return
                    Potential_BUY(data, Stop_Pips, fractal_low, "Potential")
            
            fractal_low_validity = 0

    set_15min_levels(data)

def check_limit(data, currency):
    global Pending_orders, Active_orders, Potential_orders

    if not Potential_orders.empty:
        # Close Potential Orders
        if data.name.time() == time(10,30) or data.name.time() == time(15, 30):
            for index, order in Potential_orders.iterrows():
                Potential_orders.drop(index, inplace=True)

    if not Pending_orders.empty:
                    
        # Close Pending Orders
        if data.name.time() == time(10,30) or data.name.time() == time(15, 30):
            for index, order in Pending_orders.iterrows():
                Pending_orders.drop(index, inplace=True)

        # Fill Stop Order if tagged in
        for index, order in Pending_orders.iterrows():
            if order['Buy/Sell'] == 'BUY' and data['High'] > order['Limit']:

                if check_news(data, currency):
                    if index in Pending_orders.index:
                        Pending_orders.drop(index, inplace=True)
                    return

                new_order = pd.DataFrame({
                'Sweep_DateTime': [order['Sweep_DateTime']],
                'Tag_DateTime': [data.name],
                'Buy/Sell': [order['Buy/Sell']], 
                'Limit': [order['Limit']], 
                'SL': [order['SL']],
                'TP': [order['TP']],
                'H4_High': [order['H4_High']],
                'H4_Low': [order['H4_Low']],
                'Bias': [order['Bias']],
                'Momentum': [order['Momentum']],
                'Liquidity_Level': [order['Liquidity_Level']],
                'Type': [order['Type']]})

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    Active_orders = pd.concat([Active_orders, new_order], ignore_index=True)
                if index in Pending_orders.index:
                    Pending_orders.drop(index, inplace=True)
 
            elif order['Buy/Sell'] == 'SELL' and data['Low'] < order['Limit']:
            
                if check_news(data, currency):
                    if index in Pending_orders.index:
                        Pending_orders.drop(index, inplace=True)
                    return 
                
                new_order = pd.DataFrame({
                'Sweep_DateTime': [order['Sweep_DateTime']],
                'Tag_DateTime': [data.name],
                'Buy/Sell': [order['Buy/Sell']], 
                'Limit': [order['Limit']], 
                'SL': [order['SL']],
                'TP': [order['TP']],
                'H4_High': [order['H4_High']],
                'H4_Low': [order['H4_Low']],
                'Bias': [order['Bias']],
                'Momentum': [order['Momentum']],
                'Liquidity_Level': [order['Liquidity_Level']],
                'Type': [order['Type']]})

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    Active_orders = pd.concat([Active_orders, new_order], ignore_index=True)
                if index in Pending_orders.index:           
                        Pending_orders.drop(index, inplace=True)     
                           
        # or close if SL is breached before it is tagged in
            elif order['Buy/Sell'] == 'BUY' and data['Low'] < order['SL']:
                if index in Pending_orders.index:
                    Pending_orders.drop(index, inplace=True)

            elif order['Buy/Sell'] == 'SELL' and data['High'] > order['SL']:
                if index in Pending_orders.index:
                    Pending_orders.drop(index, inplace=True)

        # print('Active Orders from check_limit_close', Active_orders)

def check_close(data, currency):
    global Active_orders, Closed_orders
    if not Active_orders.empty:           

        for index, order in Active_orders.iterrows():

            if (order['Buy/Sell'] == 'BUY' and data['Low'] <= order['SL']) or \
            (order['Buy/Sell'] == 'SELL' and data['High'] >= order['SL']):
                                
                if order['Trailed'] == 1:
                    Loss = 0
                else: Loss = -1

                new_order = pd.DataFrame({
                'Sweep_DateTime': [order['Sweep_DateTime']],
                'Buy/Sell': [order['Buy/Sell']], 
                'Limit': [order['Limit']], 
                'SL': [order['SL']],
                'TP': [order['TP']],
                'H4_High': [order['H4_High']],
                'H4_Low': [order['H4_Low']],
                'Bias': [order['Bias']],
                'Momentum': [order['Momentum']],
                'Liquidity_Level': [order['Liquidity_Level']],
                'Type': [order['Type']],
                'Tag_DateTime': [order['Tag_DateTime']],
                'Close_DateTime': [data.name],
                'Return': Loss})

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    Closed_orders = pd.concat([Closed_orders, new_order], ignore_index=True)
                Active_orders.drop(index, inplace=True)

            elif (order['Buy/Sell'] == 'BUY' and data['High'] > order['TP']) or \
            (order['Buy/Sell'] == 'SELL' and data['Low'] < order['TP']):
                
                new_order = pd.DataFrame({
                'Sweep_DateTime': [order['Sweep_DateTime']],
                'Buy/Sell': [order['Buy/Sell']], 
                'Limit': [order['Limit']], 
                'SL': [order['SL']],
                'TP': [order['TP']],
                'H4_High': [order['H4_High']],
                'H4_Low': [order['H4_Low']],
                'Bias': [order['Bias']],
                'Momentum': [order['Momentum']],
                'Liquidity_Level': [order['Liquidity_Level']],
                'Type': [order['Type']],
                'Tag_DateTime': [order['Tag_DateTime']],
                'Close_DateTime': [data.name],
                'Return': 3 })

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    Closed_orders = pd.concat([Closed_orders, new_order], ignore_index=True)
                Active_orders.drop(index, inplace=True)

            elif (order['Buy/Sell'] == 'BUY' and data['High'] > (((order['TP'])-order['Limit'])*0.66) + order['Limit']) \
                or (order['Buy/Sell'] == 'SELL' and data['Low'] < ((order['Limit'] - (order['Limit']-order['TP']))*0.66) ):
                Active_orders.at[index, 'Trailed'] = 1
                Active_orders.at[index, 'SL'] = order['Limit']

def Potential_BUY(data, Stop_Pips, Liquidity, Type):
    global Potential_orders, H4_high, H4_low, directional_bias, Momentum

    Long_StopLoss = round(data['Low'], 5) - Stop_Pips
    Long_TakeProfit = round((data['High'] + (1.5 *(data['High']-Long_StopLoss))),5)

    # Add Limit Order to open_orders to track pending orders
    new_order = pd.DataFrame({
        'Sweep_DateTime': [data.name],
        'Buy/Sell': ['BUY'], 
        'Limit': [round(data['High'], 5)], 
        'SL': [Long_StopLoss],
        'TP': [Long_TakeProfit],
        'H4_High': [H4_high],
        'H4_Low': [H4_low],
        'Bias': [directional_bias],
        'Momentum': [Momentum],
        'Liquidity_Level':[Liquidity],
        'Type': [Type]})
    

    # print("new order", new_order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        Potential_orders = pd.concat([Potential_orders, new_order], ignore_index=True)
        Potential_orders.reset_index(drop=True, inplace=True)

def Potential_SELL(data, Stop_Pips, Liquidity, Type):
    global Potential_orders, H4_high, H4_low, directional_bias, Momentum

    Short_StopLoss = round(data['High'], 5) + Stop_Pips
    Short_TakeProfit = round((data['Low'] - (1.5 *(Short_StopLoss-data['Low']))),5)

    # Add Limit Order to open_orders to track pending orders
    new_order = pd.DataFrame({
        'Sweep_DateTime': [data.name],
        'Buy/Sell': ['SELL'], 
        'Limit': [round(data['Low'], 5)], 
        'SL': [Short_StopLoss],
        'TP': [Short_TakeProfit],
        'H4_High': [H4_high],
        'H4_Low': [H4_low],
        'Bias': [directional_bias],
        'Momentum': [Momentum],
        'Liquidity_Level':[Liquidity],
        'Type': [Type]})
    
    # print("new order", new_order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        Potential_orders = pd.concat([Potential_orders, new_order], ignore_index=True)
        Potential_orders.reset_index(drop=True, inplace=True)

def BUY(data, Stop_Pips, Liquidity, Type):
    global Pending_orders, H4_high, H4_low, directional_bias, Momentum

    Long_StopLoss = round(data['Low'], 5) - Stop_Pips
    Long_TakeProfit = round((data['High'] + (1.5 *(data['High']-Long_StopLoss))),5)

    if (Long_TakeProfit >= H4_high - (0.1 * (H4_high - H4_low))) or (any(Pending_orders['Buy/Sell'] == 'BUY')) or (data['High'] - Long_StopLoss > 0.0025):
        # can add a boundary under the H4_high? like top 95% is a dead zone?
        return


    # Add Limit Order to open_orders to track pending orders
    new_order = pd.DataFrame({
        'Sweep_DateTime': [data.name],
        'Buy/Sell': ['BUY'], 
        'Limit': [round(data['High'], 5)], 
        'SL': [Long_StopLoss],
        'TP': [Long_TakeProfit],
        'H4_High': [H4_high],
        'H4_Low': [H4_low],
        'Bias': [directional_bias],
        'Momentum': [Momentum],
        'Liquidity_Level':[Liquidity],
        'Type': [Type]})

    # print("new order", new_order)

    # print("new order", new_order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        Pending_orders = pd.concat([Pending_orders, new_order], ignore_index=True)
        Pending_orders.reset_index(drop=True, inplace=True)

def SELL(data, Stop_Pips, Liquidity, Type):
    global Pending_orders, H4_high, H4_low, directional_bias, Momentum
    
    Short_StopLoss = round(data['High'], 5) + Stop_Pips
    Short_TakeProfit = round((data['Low'] - (1.5 *(Short_StopLoss-data['Low']))),5)

    if (Short_TakeProfit <= H4_low + (0.1 * (H4_high - H4_low))) or (any(Pending_orders['Buy/Sell'] == 'SELL')) or (Short_StopLoss - data['Low'] > 0.0025):
        # can add a boundary under the H4_high? like top 95% is a dead zone?
        return 
    
    # Add Limit Order to open_orders to track pending orders
    new_order = pd.DataFrame({
        'Sweep_DateTime': [data.name],
        'Buy/Sell': ['SELL'], 
        'Limit': [round(data['Low'], 5)], 
        'SL': [Short_StopLoss],
        'TP': [Short_TakeProfit],
        'H4_High': [H4_high],
        'H4_Low': [H4_low],
        'Bias': [directional_bias],
        'Momentum': [Momentum],
        'Liquidity_Level':[Liquidity],
        'Type': [Type]})

    # print("new order", new_order)

    # print("new order", new_order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        Pending_orders = pd.concat([Pending_orders, new_order], ignore_index=True)
        Pending_orders.reset_index(drop=True, inplace=True)

def check_news(data, currency_pair):
    global News

    # Extract the two currencies from the currency pair
    currency1 = currency_pair[:3]
    currency2 = currency_pair[3:]

    # Filter the news for the given currencies
    relevant_news = News[(News['Currency'] == currency1) | (News['Currency'] == currency2)]

    # Iterate through each row of the relevant news
    for index, row in relevant_news.iterrows():
        news_date = row['Date']
        news_time = row['Time']
        news_date = datetime.strptime(news_date, "%Y-%m-%d").date()
        news_time = datetime.strptime(news_time, "%H:%M").time()
        news_datetime = datetime.combine(news_date, news_time)
        input_datetime = data.name

        # Calculate the difference between the input time and the news time
        time_difference = abs((news_datetime - input_datetime).total_seconds() / 60)

        # Check if the time difference is within 15 minutes
        if time_difference <= 60:
            # print("News so no trade", data.name)
            return True  # There is a news release within 30 minutes

    return False  # No news release within 30 minutes

def run(df, Stop_Pips, currency):
    """Run the analysis. \n
    df = the 15 min price data including High, Open, Close, Low \n
    Stop_pips to indicate how many pips are added to SL \n
    currency to be a string indicating the pair being analysed eg. "GBPUSD" """
    global Order_List, News, h4_data, h4_data_fractal, H4_record, H4_last_fractal_high, H4_last_fractal_low, H4_high, H4_low, H4_high_from_fractal, H4_low_from_fractal, H4_equ,directional_bias,fractal_high, fractal_low, Min15_high, Min15_low, fractal_high_validity, fractal_low_validity,min15_record, recently_updated,Pending_orders,Active_orders,Closed_orders, m15_data_fractal,d1_data_momentum, Potential_orders

    News = pd.read_csv(r"C:\Users\hwatk\Trading\Backtesting-Code-2\News\all_news_2020-oct2023.csv")
    # News = pd.read_csv("/Users/hugowatkinson/Documents/Trading/Backtesting Code/News/all_news_2020-oct2023.csv")

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

    # df = df.resample('10T').agg({
    #     'Open': 'first',
    #     'High': 'max',
    #     'Low': 'min',
    #     'Close': 'last',
    #     'Volume': 'sum'
    # })

    # Resample to daily data 3.11 CHANGE to weekly data
    # d1_data = df.resample('1D').agg({
    #     'Open': 'first',
    #     'High': 'max',
    #     'Low': 'min',
    #     'Close': 'last',
    #     'Volume': 'sum'
    # })
    d1_data = df.resample('4H', offset='2H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })


    # Identify fractal H/L for all data (A bit of look ahead bias here?)
    m15_data_fractal_1 = identify_fractals(df)
    m15_data_fractal = identify_bar(m15_data_fractal_1)
    h4_data_fractal_1 = identify_fractals(h4_data)
    h4_data_fractal = identify_bar(h4_data_fractal_1)
    h4_data_fractal['Directional_bias'] = None
    d1_data_fractal_1 = identify_fractals(d1_data)
    d1_data_fractal = identify_bar(d1_data_fractal_1)    

    # Identify daily momentum
    d1_data_momentum = update_daily_momentum(d1_data_fractal)

    # price_data_with_atr = calculate_atr(h4_data_fractal)
    # price_data_with_atr.to_csv(os.path.join(Path,'ATR.csv'))

    # Create the H4 record 
    H4_record = pd.DataFrame(columns=['Datetime','H4_high', 'H4_low','Last_Fractal_High','Last_Fractal_Low', 'Directional_bias','Bar_High', 'Bar_Low'])

    # H4 Variables
    H4_last_fractal_high = 0
    H4_last_fractal_low = 0
    H4_high = 0
    H4_low = 0
    H4_high_from_fractal = False
    H4_low_from_fractal = False
    H4_equ = 0
    # directional_bias = 0

    # m15 Variables
    fractal_high, fractal_low, Min15_high, Min15_low, fractal_high_validity, fractal_low_validity = 0,0,0,0,0,0
    min15_record = pd.DataFrame(columns=['Datetime','M15_high', 'M15_low'])
    recently_updated = False
    Pending_orders = pd.DataFrame(columns = ['Sweep_DateTime', 'Buy/Sell', 'Limit', 'SL', 'TP', 'H4_High', 'H4_Low', 'Bias', 'Momentum', 'Type'])
    Potential_orders = pd.DataFrame(columns = ['Sweep_DateTime', 'Buy/Sell', 'Limit', 'SL', 'TP', 'H4_High', 'H4_Low', 'Bias', 'Momentum', 'Liquidity_Level', 'Type'])
    Active_orders = pd.DataFrame(columns = ['Sweep_DateTime', 'Buy/Sell', 'Limit', 'SL', 'TP', 'H4_High', 'H4_Low', 'Bias', 'Momentum', 'Liquidity_Level', 'Tag_DateTime', 'Type', 'Trailed'])
    Closed_orders = pd.DataFrame(columns = ['Sweep_DateTime', 'Buy/Sell', 'Limit', 'SL', 'TP', 'H4_High', 'H4_Low', 'Bias', 'Momentum', 'Liquidity_Level', 'Tag_DateTime', 'Type','Close_DateTime','Return'])

    #### Main Loop ###
    for  index, bar in tqdm(m15_data_fractal.iterrows()):
        check_limit(bar, currency)
        check_close(bar, currency)
        check_15min_entry(bar, Stop_Pips, 5)
        if index in h4_data_fractal.index:
            h4_row = h4_data_fractal.loc[index]
            update_H4_range(h4_row)
            update_h4_fractals(h4_row)
            directional_bias = h4_data_fractal.loc[index, 'Directional_bias']


    # H4_record.to_csv(os.path.join(Path, 'H4_Record.csv'), index=False)
    # h4_data_fractal.to_csv(os.path.join(Path, 'PriceData.csv'))


    # Closed orders to Order_Lits
    Closed_orders['Currency Pair'] = currency 

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        Order_List = pd.concat([Order_List, Closed_orders], ignore_index = False)
    # Rename index
    # Order_List = Order_List.rename_axis('Close Datetime')
    # Drop empty column
    # Order_List = Order_List.drop('Close Datetime', axis=1, errors='ignore')

    # high_prob_rate, directional_bias_success_rate = process_results(Closed_orders, H4_record, d1_data_momentum)
    # print(f"High Probability Success Rate: {high_prob_rate:.2f}%")
    # print(f"4H Directional Bias Success Rate: {directional_bias_success_rate:.2f}%")

#####################################################################################################################
Path = r'C:\Users\hwatk\Trading\Backtesting-Code-2\Active\Output'
for filename in os.listdir(Path):
    file_path = os.path.join(Path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

Order_List = pd.DataFrame(columns = ['Sweep_DateTime', 'Buy/Sell', 'Limit', 'SL', 'TP', 'H4_High', 'H4_Low', 'Bias', 'Momentum', 'Liquidity_Level', 'Tag_DateTime', 'Close_DateTime','Return', 'Type', 'Currency Pair'])

print('EURUSD analysis')
df = pd.read_csv("C:\\Users\\hwatk\\Trading\\Backtesting-Code-2\\Historical Data\\eurusd-m5-bid-2020-09-16-2023-09-16.csv")
# df = pd.read_csv("C:\\Users\\hwatk\\Trading\\Backtesting-Code-2\\Historical Data\\eurusd-m15-bid-2020-09-16-2023-09-16.csv")
df = df.iloc[(-90000):]
run(df, 0.0003, "EURUSD")

if not os.path.exists(Path):
    os.makedirs(Path)
Order_List = Order_List.sort_index()
Order_List.to_csv(os.path.join(Path, 'Order_List.csv'), index=False)

ProcessAll.Process_all()
# Note EURUSD needs to be changed if alternative currency. Line 86