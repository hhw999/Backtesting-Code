import pandas as pd
import numpy as np
from pandas import Timedelta
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import matplotlib.dates as mdates
from datetime import datetime, time, timedelta
from pytz import UTC  # Import the UTC timezone from pytz library
import os
import PlotTrades
    
def Process_all():
    Path = '/Users/hugowatkinson/Documents/Trading/Backtesting Code/Active/Output/'
    if not os.path.exists(Path):
        os.makedirs(Path, exist_ok=True)

    # Sort by datetime index
    Order_List = pd.read_csv("/Users/hugowatkinson/Documents/Trading/Backtesting Code/Active/Output/Order_List.csv")


    Order_List.set_index('Close_DateTime', inplace=True)
    Order_List.index = pd.to_datetime(Order_List.index)

    # print(Order_List.to_string())

    def max_consecutive_negatives(series):
        max_drawdown = 0
        current_drawdown = 0
        for value in series:
            if value == -1:
                current_drawdown += 1
                max_drawdown = max(max_drawdown, current_drawdown)
            else:
                current_drawdown = 0
        return max_drawdown

    # Calculate percentage positive months
    Pos_Month_rate = round((len(Order_List[Order_List['Return'] > 0]) / len(Order_List)) * 100, 4)

    # Group by currency pair and month, then calculate total return, drawdown, and number of trades
    monthly_currency_breakdown = Order_List.groupby([pd.Grouper(freq='M'), 'Currency Pair']).agg(
        total_return=pd.NamedAgg(column='Return', aggfunc='sum'),
        drawdown=pd.NamedAgg(column='Return', aggfunc=max_consecutive_negatives),
        num_trades=pd.NamedAgg(column='Return', aggfunc='count')
    ).reset_index()

    # Save total monthly breakdown for each currency pair to CSV
    for currency_pair in Order_List['Currency Pair'].unique():
        currency_df = monthly_currency_breakdown[monthly_currency_breakdown['Currency Pair'] == currency_pair]
        
        currency_df.to_csv(os.path.join(Path,f'MonthlyBreakdown_{currency_pair}.csv'), index=False)

    # Group by month, then calculate total return, drawdown, and number of trades
    monthly_breakdown = Order_List.groupby([pd.Grouper(freq='M')]).agg(
        total_return=pd.NamedAgg(column='Return', aggfunc='sum'),
        drawdown=pd.NamedAgg(column='Return', aggfunc=max_consecutive_negatives),
        num_trades=pd.NamedAgg(column='Return', aggfunc='count')).reset_index()

    # Save total monthly breakdown to CSV
    monthly_breakdown.to_csv(os.path.join(Path,f'TotalMonthlyBreakdown.csv'), index=False)

    # Calculate total returns, max drawdown, and number of trades
    total_return = Order_List['Return'].sum()
    max_drawdown = max_consecutive_negatives(Order_List['Return'])
    num_trades = len(Order_List)

    # Save total results to CSV
    total_results = pd.DataFrame({
        'Total Return': [total_return],
        'Max Drawdown': [max_drawdown],
        'Number of Trades': [num_trades],
    })
    total_results.to_csv(os.path.join(Path,f'TotalResults.csv'), index=False)

    # Print results
    print("Total Return:", total_return)
    print("Max Drrawdown:", max_drawdown)
    print('Percentage positive months:', Pos_Month_rate)
    
    # print("Total Monthly Breakdown:\n", monthly_currency_breakdown.to_string())
    # Plot equity curve and output monthly account results
    equity_curve(Order_List)

    pair = 'eurusd'
    order_list_file = '/Users/hugowatkinson/Documents/Trading/Backtesting Code/Active/Output/Order_List.csv'
    price_data_file = f'/Users/hugowatkinson/Documents/Trading/Historical Data/{pair}-m15-bid-2020-09-16-2023-09-16.csv'
    output_folder = '/Users/hugowatkinson/Documents/Trading/Backtesting Code/Active/Output/output_plots'
    PlotTrades.process_trades(order_list_file, price_data_file, output_folder, 50)

def equity_curve(Order_List):

    start_equity = 100000
    profit_split = 0.8

    Total_Pay_Out = 0
    Accounts_Bought = 1
    Account_Cost = 300
    Account_Record = pd.DataFrame(columns=['Date', 'Payout', 'Deficit'])

    Path = '/Users/hugowatkinson/Documents/Trading/Backtesting Code/Active/Output/'
    
    # print('order list \n', Order_List.to_string())

    # Step 0: Reset index, reindex to include the first of every month, and then set index back
    start_date = Order_List.index.min()
    end_date = Order_List.index.max()

    duplicates = Order_List.index.duplicated(keep='first')

    # Remove duplicates
    Order_List = Order_List[~duplicates]

    # Generate new index for the first of each month within the range
    new_index = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Create a new index that combines the old index and the new index
    new_index = Order_List.index.union(new_index)

    # Reindex the dataframe with the combined index
    Order_List = Order_List.reindex(new_index)

    # Sort the index
    Order_List = Order_List.sort_index()

    # Set up an equity variable, and set 0 to start_equity
    Order_List['Equity'] = 90000
    Order_List.at[Order_List.index.min(), 'Equity'] = start_equity
    Order_List['Return'] = Order_List['Return'].fillna(0)

    # print('Order list \n', Order_List.to_string())

    last_processed_month = None

    # Loop through orders and update Equity
    for i in range(1, len(Order_List)):

        if pd.isna(Order_List['Return'].iloc[i]) or ((Order_List['Return'].iloc[i]) == 0) :
            # print("IS NA", i)
            Order_List.iloc[i, Order_List.columns.get_loc('Equity')] = Order_List['Equity'].iloc[i-1]
        elif Order_List['Equity'].iloc[i-1] < (start_equity * 0.9):
            print("ACCOUNT LOST")
            Accounts_Bought += 1
            Order_List['Equity'].iloc[i] = start_equity
            # break
        elif Order_List['Return'].iloc[i] == 3:
            # print("return3", i)
            Order_List.iloc[i, Order_List.columns.get_loc('Equity')] = Order_List['Equity'].iloc[i-1] * 1.029
        elif Order_List['Return'].iloc[i] == -1:
            # print("return-1", i)
            Order_List.iloc[i, Order_List.columns.get_loc('Equity')] = Order_List['Equity'].iloc[i-1] * 0.989

        current_month = Order_List.index[i].month

        # Update equity at end of month
        if Order_List.index[i].day == 1 and current_month != last_processed_month:
            last_processed_month = current_month  # update the last processed month
        
            if Order_List['Equity'].iloc[i-1] > start_equity:
                payout = ((round((Order_List['Equity'].iloc[i-1])-start_equity,2))* profit_split)
                Total_Pay_Out += payout
                new_row = {'Date': Order_List.index[i], 'Payout': payout, 'Deficit': np.nan}
                Account_Record = pd.concat([Account_Record, pd.DataFrame([new_row])], ignore_index=True)
                Order_List.iloc[i, Order_List.columns.get_loc('Equity')] = start_equity

            elif Order_List['Equity'].iloc[i-1] < start_equity:
                defecit = (round(start_equity-(Order_List['Equity'].iloc[i-1]),2))
                new_row = {'Date': Order_List.index[i], 'Payout': np.nan, 'Deficit': defecit}
                Account_Record = pd.concat([Account_Record, pd.DataFrame([new_row])], ignore_index=True)

            elif Order_List['Equity'].iloc[i-1] == start_equity:
                print("Breakeven month")


    print("Total payouts: £", Total_Pay_Out)
    print("Total accounts costs:", (Accounts_Bought*Account_Cost))
    print("Total takeaway: £", (Total_Pay_Out - (Accounts_Bought*Account_Cost)))
    # print("order list equity \n", Order_List['Equity'].to_string())
    # print("Account record \n", Account_Record)
    Account_Record.to_csv(os.path.join(Path,f"Account_Record.csv"))

    # Plot curve
    plt.plot(Order_List.index, Order_List['Equity'], label='Equity')
    plt.axhline(y=start_equity, color='b', linestyle='-', label='£100,000')
    # plt.axhline(y=(start_equity*0.9), color='r', linestyle='-', label='£90,000')
    plt.xlabel('Time')
    plt.ylabel('Equity (£)')
    plt.title('Equity Curve')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    Process_all()   