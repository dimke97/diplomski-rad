import pandas as pd
import yfinance as yf

import sys
import os

import bs4 as bs
from datetime import date

import requests


def get_response(url):
    response = requests.get(url)
    response.raise_for_status()  # raises exception when not a 2xx response
    if response.status_code != 204:
        return response.json()


def get_exchange_info():
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/exchangeInfo"
    return get_response(base_url + endpoint)


def create_symbols_list(filter="USDT"):
    # rows = []
    info = get_exchange_info()
    pairs_data = info["symbols"]
    full_data_dic = {s["symbol"]: s for s in pairs_data if filter in s["symbol"]}
    return full_data_dic.keys()


def collect_tickers(start_date="2013-01-01", end_date=None):
    tickers = create_symbols_list("USDT")
    tickers = [ticker[:-1] for ticker in tickers]
    # tickers = [ticker[:3] + "-" + ticker[3:] for ticker in tickers]
    tickers_all = []
    for ticker in tickers:
        if len(ticker) == 6:
            ticker_temp = ticker[:3] + "-" + ticker[3:]
            tickers_all.append(ticker_temp)
        elif len(ticker) == 7:
            ticker_temp = ticker[:4] + "-" + ticker[4:]
            tickers_all.append(ticker_temp)
        else:
            print(ticker)
            print("Neki drugi ticker je u pitanju.")

    coins = yf.download(tickers=tickers_all, start=start_date, end=end_date)
    coins = coins.loc[:, "Adj Close"]
    coins = coins.dropna(axis=1, how="all")

    coins.columns = [col.replace("-USD", "") for col in coins.columns]

    columns_no_na = coins.iloc[-1].isna()
    columns_no_na = columns_no_na.iloc[columns_no_na.to_numpy().nonzero()].index

    coins = coins.drop(columns=columns_no_na)
    coins = coins.drop(
        columns=[
            "AUD",
            "UNI",
            "VIC",
            "APE",
            "TIA",
            "NBS",
            "JUP",
            "MEME",
            "BTT",
            "SHIB",
            "BONK",
        ]
    )

    return coins
    # coins.to_csv("data/coins.csv")


def coin_exists():
    if os.path.exists("data/coins.csv"):
        return True
    else:
        False


def max_date(filepath="data/coins.csv"):
    df = pd.read_csv(filepath)
    return pd.to_datetime(df["Date"].max()).date()
