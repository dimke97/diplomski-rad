import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date


date_today = date.today()


def from_data_file():
    url = "data/coins.csv"
    return pd.read_csv(url, index_col=[0], parse_dates=[0])


def ann_risk_return(returns_df):
    returns = returns_df.pct_change()
    df = returns.agg(["mean", "std"]).T
    df.columns = ["Return", "Risk"]
    df.Return = df.Return * 365
    df.Risk = df.Risk * np.sqrt(365)

    return df


def capm_model(coin_data, daily_ret, rf=0.0388):
    risk_free_return = rf
    risk_free_risk = 0
    rf = [risk_free_return, risk_free_risk]

    df = ann_risk_return(coin_data)

    df["Sharpe"] = (df["Return"].sub(rf[0])) / df["Risk"]
    cov = daily_ret.cov() * 365
    # print(cov)

    df["SystRisk_var"] = cov.iloc[:, -1]
    df["TotalRisk_var"] = np.power(df.Risk, 2)
    df["UnsystRisk_var"] = df["TotalRisk_var"].sub(df["SystRisk_var"])
    df["beta"] = df.SystRisk_var / df.loc["BTC", "SystRisk_var"]
    df["capm_ret"] = rf[0] + (df.loc["BTC", "Return"] - rf[0]) * df.beta
    df["alpha"] = df.Return - df.capm_ret
    df["Er"] = rf[0] + df["beta"] * (df.loc["BTC", "Return"])

    df = df[df["beta"] >= 0]

    return df


def calculate():
    coins = from_data_file()

    an_returns = ann_risk_return(coins)
    an_returns.to_csv("data/return_to_risk.csv")

    raw_returns = coins.pct_change()
    raw_returns.to_csv("data/daily_returns.csv")

    capm = capm_model(coins, raw_returns)
    capm.to_csv("data/capm_data.csv")
