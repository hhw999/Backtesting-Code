import requests
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import plotly.graph_objs as go

# av_apiKey = "RLNMS8H0P6TRME5J"
av_apiKey = "QKOBDHQEO1I111IL"

# function that will get the 1H stats for a stock indicated by paramter `symbol`
#    then return it as a pandas DataFrame `df`
#    def get_stock_data_daily(symbol, interval):
#     # url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=60min&apikey={av_apiKey}&datatype=csv"
#     slice = "year1month1"
#     url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={symbol}&interval={interval}&slice={slice}&apikey={av_apiKey}&datatype=csv"
#     response = requests.get(url)
#     csv_data = response.content
#     df = pd.read_csv(io.BytesIO(csv_data))
#     return df


# H1_data = get_stock_data_daily(stock_symbol, chart)
# print(H1_data)


symbol = "GBPJPY"

#Download d1, h4 and h1 chart data for a year
d1_data = yf.download(tickers = 'GBPJPY=X' ,start="2022-06-01", end="2023-5-30", interval = '1d')
h1_data = yf.download(tickers = 'GBPJPY=X' ,start="2022-06-01", end="2023-5-30", interval = '1h')
# 1h to 4h
df_agg = h1_data.groupby(pd.Grouper(freq='4H')).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last","Adj Close": "last"})
# Remove the NaN rows
h4_data = df_agg.dropna(how='all')
# Label the dataframe columns 
h4_data.columns = ["open", "high", "low", "close", "Adj Close"]


# Create a trade log. To include buy/seel, 1h entry data, SL, TP1 (breakeven), TP2, and close data. Maybe have an active and closed list? 



# Find a structure defining code. swing highs and lows


# Trend check and MTF trend direction



# Create an order block log. To be defined by structure on h4, refined to h1. bullish or bearish (might be good to see if this matters)? Delete when invalidated. 



# Entry on engulfing out of order block when MTF in line with engulfing



# Track trade, If SL, TP1, TP2 record close. 






############### 
# Chart

# #declare figure
# fig = go.Figure()
# #Candlestick
# fig.add_trace(go.Candlestick(x=data.index,
# open=data['Open'],
# high=data['High'],
# low=data['Low'],
# close=data['Close'], name = 'market data'))
# # Add titles
# fig.update_layout(title="{symbol}")

# #Show
# fig.show()

