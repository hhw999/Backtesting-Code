import pandas as pd
from datetime import datetime

def filter_and_append_csv(file_list, start_date, end_date, output_file):
    # Convert start and end date strings to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Open output file in append mode
    with open(output_file, 'a') as f:
        for file in file_list:
            # Read CSV file without headers
            df = pd.read_csv(file, header=None)

            # Convert second column (index 0) to datetime
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.iloc[:, 0] = df.iloc[:, 0].dt.date

            # Filter rows based on date range and 'H' in 4th column (index 3)
            filtered_df = df[(df.iloc[:, 0] >= start_date) & (df.iloc[:, 0] <= end_date) & (df.iloc[:, 3] == 'H')]

            df_filtered = df_filtered.drop("Column4", axis=1)

            # Append filtered rows to output file
            filtered_df.to_csv(f, header=False, index=False, mode='a')


if __name__ == "__main__":
    # List of CSV files to process
    files = ['/Users/hugowatkinson/Documents/Trading/Backtesting Code/News/FF calendar news events 2020.csv', \
            '/Users/hugowatkinson/Documents/Trading/Backtesting Code/News/FF calendar news events 2021.csv', \
            '/Users/hugowatkinson/Documents/Trading/Backtesting Code/News/FF calendar news events 2022.csv', \
            '/Users/hugowatkinson/Documents/Trading/Backtesting Code/News/FF calendar news events 2023.csv']

    # Start and end date for filtering
    start_date = "2020-01-01"
    end_date = "2023-10-31"

    # Output file
    output_file = "/Users/hugowatkinson/Documents/Trading/Backtesting Code/News/all_news_2020-oct2023.csv"

    # Call the function to filter and append CSV files
    filter_and_append_csv(files, start_date, end_date, output_file)
