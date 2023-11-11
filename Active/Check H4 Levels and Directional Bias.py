# Check H4 Levels and Directional Bias

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
# import os
# import shutil
# import ProcessAll
# import warnings
# import Backtesting3_0


# Read the "PriceData.csv" file
price_data = pd.read_csv("/Users/hugowatkinson/Documents/Trading/Backtesting Code/Active/Output/PriceData.csv", parse_dates=True, index_col="timestamp")

# Read the "H4_Record.csv" file
h4_record = pd.read_csv("/Users/hugowatkinson/Documents/Trading/Backtesting Code/Active/Output/H4_Record.csv", parse_dates=True, index_col="Datetime")

# Check for and remove duplicate timestamps in "H4_Record"
h4_record = h4_record[~h4_record.index.duplicated(keep='first')]

# Ensure the timestamps are aligned
price_data = price_data.reindex(h4_record.index, method='ffill')

print(h4_record)
print(price_data)

# Create a figure and axis
fig, axlist = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Create a candlestick chart using mplfinance in the top subplot
mpf.plot(price_data, type='candle', style='yahoo', ax=axlist[0])

# Overlay H4_high and H4_low on the same axis
axlist[0].plot(h4_record.index, h4_record["H4_high"], label="H4 High", color="green")
axlist[0].plot(h4_record.index, h4_record["H4_low"], label="H4 Low", color="red")

# Customize the top subplot
axlist[0].legend()
axlist[0].grid()
axlist[0].set_title("Candlestick Chart")

# Create a subplot for H4_high and H4_low data in the bottom subplot
axlist[1].plot(h4_record.index, h4_record["H4_high"], label="H4 High", color="green")
axlist[1].plot(h4_record.index, h4_record["H4_low"], label="H4 Low", color="red")
axlist[1].legend()
axlist[1].grid()

# Display the combined chart
plt.tight_layout()
plt.show()