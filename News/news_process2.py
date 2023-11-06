import pandas as pd
from datetime import datetime

df = pd.read_csv("/Users/hugowatkinson/Documents/Trading/Backtesting Code/News/all_news_2020-oct2023_w.headers.csv")

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.date

output_file = "/Users/hugowatkinson/Documents/Trading/Backtesting Code/News/all_news_2020-oct2023.csv"

df.to_csv(output_file)