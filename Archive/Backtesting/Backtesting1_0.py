import requests
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import plotly.graph_objs as go


#Download d1, h4 and h1 chart data for a year
d1_data = yf.download(tickers = 'GBPUSD=X' ,start="2022-06-01", end="2023-5-30", interval = '1d')
print(d1_data)

# h1_data = yf.download(tickers = 'GBPJPY=X' ,start="2022-06-01", end="2023-5-30", interval = '1h')
# # 1h to 4h
# df_agg = h1_data.groupby(pd.Grouper(freq='4H')).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last","Adj Close": "last"})
# # Remove the NaN rows
# h4_data = df_agg.dropna(how='all')
# # Label the dataframe columns 
# h4_data.columns = ["open", "high", "low", "close", "Adj Close"]

# Create a trade log. To include buy/seel, 1h entry data, SL, TP1 (breakeven), TP2, and close data. Maybe have an active and closed list? 
orders_open = pd.DataFrame([],columns=['OrderNo', 'Buy/Sell', 'Open Datetime', 'Open Open', 'Open High', 'Open Low', 'Open Close', 'StopLoss', 'TakeProfit', 'BreakEven','Open Bar'])
orders_closed = pd.DataFrame([],columns=['OrderNo', 'Buy/Sell', 'Open Datetime', 'Open Open', 'Open High', 'Open Low', 'Open Close', 'StopLoss', 'TakeProfit', 'BreakEven''Open Bar',
                                         'Close Datetime', 'Close Open', 'Close High', 'Close Low', 'Close Close', 'Close Bar'])

# Find a structure defining code. swing highs and lows

# Trend check and MTF trend direction

# Create an order block log. To be defined by structure on h4, refined to h1. bullish or bearish (might be good to see if this matters)? Delete when invalidated. 

# Iterate through data
orderNo = 1
for index, row in d1_data.iterrows():

    # Define bar bull/bear
    if d1_data["Open"][index] > d1_data["Close"][index]: 
        d1_data.at[index,"Bar"] = "Bull"
    else: d1_data.at[index,"Bar"] = "Bear"

    print('order check')

    #If SL, TP1, TP2 record and close. 
    for i, row in orders_open.iterrows(): 
            # Close and log if TP
        if orders_open["Buy/Sell"][i] == 'Buy':
            if d1_data["High"][index] > orders_open['TakeProfit'][i]:
                # Add closed order to log
                closing = {
                            'OrderNo.': orders_open['OrderNo'][i], 
                            'Buy/Sell': orders_open['Buy/Sell'][i],
                            'Open Datetime': orders_open['Open Datetime'][i], 
                            'Open Open': orders_open['Open Open'][i], 
                            'Open High': orders_open['Open High'][i], 
                            'Open Low': orders_open['Open Low'][i],
                            'Open Close': orders_open['Open Close'][i],
                            'StopLoss': orders_open['StopLoss'][i],
                            'TakeProfit': orders_open['TakeProfit'][i],
                            'BreakEven': orders_open['BreakEven'][i],
                            'Open Bar': orders_open['Bar'][i],
                            'Close Datetime': index,
                            'Close Open': d1_data["Open"][index], 
                            'Close High': d1_data["High"][index], 
                            'Close Low': d1_data["Low"][index], 
                            'Close Close': d1_data["Close"][index],
                            'Close Bar': d1_data['Bar'][i],}    
                orders_closed = pd.concat([orders_closed, pd.DataFrame([closing])], ignore_index=True)
                # Drop open order
                orders_open = orders_open.drop([i])
                print("Take profit")

            if d1_data["Low"][index] < orders_open['StopLoss'][i]:
                # Add closed order to log
                closing = {
                            'OrderNo.': orders_open['OrderNo'][i], 
                            'Buy/Sell': orders_open['Buy/Sell'][i],
                            'Open Datetime': orders_open['Open Datetime'][i], 
                            'Open Open': orders_open['Open Open'][i], 
                            'Open High': orders_open['Open High'][i], 
                            'Open Low': orders_open['Open Low'][i],
                            'Open Close': orders_open['Open Close'][i],
                            'StopLoss': orders_open['StopLoss'][i],
                            'TakeProfit': orders_open['TakeProfit'][i],
                            'BreakEven': orders_open['BreakEven'][i],
                            'Open Bar': orders_open['Bar'][i],
                            'Close Datetime': index,
                            'Close Open': d1_data["Open"][index], 
                            'Close High': d1_data["High"][index], 
                            'Close Low': d1_data["Low"][index], 
                            'Close Close': d1_data["Close"][index],
                            'Close Bar': d1_data['Bar'][i],}      
                orders_closed = pd.concat([orders_closed, pd.DataFrame([closing])], ignore_index=True)
                # Drop open order
                orders_open = orders_open.drop([i])
                print('STOP LOSS')

        elif d1_data["Buy/Sell"][index] == "Sell":
            if d1_data["Low"][index] < orders_open['TakeProfit'][i]:
                # Add closed order to log
                closing = {
                             'OrderNo.': orders_open['OrderNo'][i], 
                            'Buy/Sell': orders_open['Buy/Sell'][i],
                            'Open Datetime': orders_open['Open Datetime'][i], 
                            'Open Open': orders_open['Open Open'][i], 
                            'Open High': orders_open['Open High'][i], 
                            'Open Low': orders_open['Open Low'][i],
                            'Open Close': orders_open['Open Close'][i],
                            'StopLoss': orders_open['StopLoss'][i],
                            'TakeProfit': orders_open['TakeProfit'][i],
                            'BreakEven': orders_open['BreakEven'][i],
                            'Open Bar': orders_open['Bar'][i],
                            'Close Datetime': index,
                            'Close Open': d1_data["Open"][index], 
                            'Close High': d1_data["High"][index], 
                            'Close Low': d1_data["Low"][index], 
                            'Close Close': d1_data["Close"][index],
                            'Close Bar': d1_data['Bar'][i],}     
                orders_closed = pd.concat([orders_closed, pd.DataFrame([closing])], ignore_index=True)
                # Drop open order
                orders_open = orders_open.drop([i])

            if d1_data["High"][index] > orders_open['StopLoss'][i]:
                # Add closed order to log
                closing = {
                            'OrderNo.': orders_open['OrderNo'][i], 
                            'Buy/Sell': orders_open['Buy/Sell'][i],
                            'Open Datetime': orders_open['Open Datetime'][i], 
                            'Open Open': orders_open['Open Open'][i], 
                            'Open High': orders_open['Open High'][i], 
                            'Open Low': orders_open['Open Low'][i],
                            'Open Close': orders_open['Open Close'][i],
                            'StopLoss': orders_open['StopLoss'][i],
                            'TakeProfit': orders_open['TakeProfit'][i],
                            'BreakEven': orders_open['BreakEven'][i],
                            'Open Bar': orders_open['Bar'][i],
                            'Close Datetime': index,
                            'Close Open': d1_data["Open"][index], 
                            'Close High': d1_data["High"][index], 
                            'Close Low': d1_data["Low"][index], 
                            'Close Close': d1_data["Close"][index],
                            'Close Bar': d1_data['Bar'][i],}      
                orders_closed = pd.concat([orders_closed, pd.DataFrame([closing])], ignore_index=True)
                # Drop open order
                orders_open = orders_open.drop([i])
        
        else: print("no close")


    # Entry on engulfing out of order block when MTF in line with engulfing
    # Entry
    if d1_data["Low"][index] < 1.1:
        order = {'OrderNo': orderNo, 
                 'Buy/Sell': 'Buy',
                 'Open Datetime': index, 
                 'Open Open': d1_data["Open"][index], 
                 'Open High': d1_data["High"][index], 
                 'Open Low': d1_data["Low"][index],
                 'Open Close': d1_data["Close"][index],
                 'StopLoss': d1_data["Close"][index]-0.010,
                 'TakeProfit': d1_data["Close"][index]+0.030,
                 'BreakEven':d1_data["Close"][index]+0.010,
                 'Bar': d1_data["Bar"][index] }
        orders_open = pd.concat([orders_open, pd.DataFrame([order])], ignore_index=True)
        print('orders open')
        print(orders_open)
        orderNo = orderNo + 1

print('d1 data')
print(d1_data)
print('orders closed')
print(orders_closed)





###############
#declare Candlestick figure
fig = go.Figure()
fig.add_trace(go.Candlestick(x=d1_data.index,
    open=d1_data['Open'],
    high=d1_data['High'],
    low=d1_data['Low'],
    close=d1_data['Close'], 
    name = 'market data'))
fig.update_layout(title="symbol")
fig.show()