import streamlit as st
from datetime import date, timedelta
import pandas as pd

from src.get_crypto_data import collect_tickers, coin_exists, max_date
from src.calculations import calculate

import warnings

warnings.filterwarnings("ignore")

start_date = date(2013, 1, 1)
end_date = date.today()

is_coins_exists = coin_exists()
if is_coins_exists:
    start_date = max_date() + timedelta(days=1)
    if start_date < end_date:
        old_coins = pd.read_csv("data/coins.csv")
        new_coins = collect_tickers(start_date, end_date)
        coins = pd.concat([old_coins, new_coins])
        coins.to_csv("data/coins.csv")
    else:
        print("Vec je pustena skripta danas.")
else:
    coins = collect_tickers(start_date, end_date)
    coins.to_csv("data/coins.csv")

calculate()

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
