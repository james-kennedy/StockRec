import csv
import datetime
import json
import os
import sys

from ratelimit import limits, sleep_and_retry
import yfinance as yf


# list of the tickers for downloading data - defaults to S&P 500 ETF (SPY)
SPY_TSV_LIST = os.path.join(os.getcwd(), "spy500list.tsv")

# set output data file location
OUTPUT_FILE = os.path.join(os.getcwd(), "spy_data.json")

# set variables to limit rate (not needed for small datasets)
RATE_LIMIT_TIME = 60 * 60   # time period for rate limit in seconds (currently 1 hour)
RATE_LIMIT_CALLS = 2000     # 2000 calls allowed per hour by YFin using IP auth


# only allow up to the ratelimit - sleep and retry if rate is exceeded
@sleep_and_retry
@limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_TIME)
def get_individual_stock_data(ticker):
    """
    :param ticker:  (string) stock ticker code
    :return:        dict object of stock data, or empty dict if no data found
    """
    this_stock = yf.Ticker(ticker)
    if this_stock:
        print("Got data for {}".format(ticker))
        print(this_stock.info)
        return this_stock.info
    print("WARNING: no info found for ticker {}".format(ticker))
    return {}


def get_all_stock_data(ticker_list):
    """
    :param ticker_list: (list of string) tickers for all stocks to fetch
    Takes the ticker_list, collects data from Yahoo Finance for each ticker, adds to json object
    Final result is dumped to .json file
    """
    time_started = datetime.datetime.now()
    print("Data collection started at {}".format(time_started))

    # create the json object for collating all data and metadata
    spy_data = {
        "time_started": str(time_started),
        "stock_data": {}
    }

    # fetch the data for each ticker and add to the dictionary
    for ticker in ticker_list:
        # prevent duplicates
        if ticker not in spy_data["stock_data"]:
            stock_data = get_individual_stock_data(ticker)
            if stock_data:
                spy_data["stock_data"][ticker] = stock_data
    
    # report back the time for processing and add it to the metadata
    time_finished = datetime.datetime.now()
    spy_data["time_finished"] = str(time_finished)
    time_elapsed = (time_finished - time_started).total_seconds() / 60.0
    print("Data collection finished at {}".format(time_finished))
    print("Took approximately {} minutes".format(time_elapsed))

    # dump data to json file
    print("Starting file write")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(spy_data, f, ensure_ascii=False, indent=4)
    print("File write completed")


def get_list_from_tsv(tsv_file_path, ticker_column=0, headers=True):
    """
    :param tsv_file_path:   (string) file path for tsv file
    :param ticker_column:   (int) index of column in data for the ticker string
    :param headers:         (bool) whether first row of TSV are headers
    :return:                (list of string) list from each entry in provided TSV
    Extract a list of strings from a given TSV path and return data
    """
    # check the TSV is a valid file path
    if os.path.isfile(tsv_file_path):
        # open TSV and read values
        with open(tsv_file_path) as tsv_input:
            raw_tsv = csv.reader(tsv_input, delimiter='\t')
            if headers:
                next(raw_tsv, None)  # skip the headers
            return [x[ticker_column] for x in raw_tsv]
    return []


def main():
    ticker_list = get_list_from_tsv(SPY_TSV_LIST)

    # if no data is found, end execution
    if not ticker_list:
        sys.exit("Invalid file, or no ticker list provided in file {}".format(SPY_TSV_LIST))
    
    get_all_stock_data(ticker_list)


if __name__ == "__main__":
    main()
