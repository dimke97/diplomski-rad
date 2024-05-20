import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from urllib.error import URLError
import numpy as np

st.set_page_config(page_title="Informacije o kriptovaluti", page_icon="🌍")

st.markdown("# Informacije o kriptovaluti")
st.sidebar.header("Izaberite coin")


@st.cache_data
def get_all_coins():
    url = "data/coins.csv"
    return pd.read_csv(url, index_col=[0], parse_dates=[0])


@st.cache_data
def get_return_data():
    url = "data/daily_returns.csv"
    return pd.read_csv(url, index_col=[0], parse_dates=[0])


@st.cache_data
def get_capm_data():
    url = "data/capm_data.csv"
    return pd.read_csv(url, index_col=[0], parse_dates=[0])


coins = get_all_coins()
return_data = get_return_data()
capm_data = get_capm_data()

list_of_coins = list(coins.columns)

crypto_option = st.multiselect(
    "Koju kriptovalutu zelite da analizirate?", list_of_coins, ["BTC"]
)

# Vizuelizacija prvog chart-a
if len(crypto_option) == 1:
    option_coin = coins.loc[:, crypto_option].dropna()
    st.line_chart(option_coin)
else:
    option_coin = coins.loc[:, crypto_option].dropna()
    norm_coin = option_coin.div(option_coin.iloc[0]).mul(100)

    st.line_chart(norm_coin)


ret = return_data[crypto_option].pct_change().dropna()
an_ret = capm_data.loc[crypto_option]


st.scatter_chart(an_ret.reset_index(), x="Risk", y="Return", color="index")

# Stopa prinosa kroz mesece
ret_chart = ret.resample("M").mean()
fig, ax = plt.subplots(figsize=(16, 8))
for i, co in enumerate(crypto_option):
    # Select the specific column from ret_chart for the y-axis
    datumi = [str(d)[:10] for d in ret_chart.index]
    sns.barplot(
        x=datumi,
        y=ret_chart[co],
        data=ret_chart,
        color=sns.color_palette("RdYlGn")[i],
        label=co,
        alpha=0.5,
    )
fig.tight_layout()
ax.set_title("Stope prinosa kroz mesece", fontsize=16)
ax.set_xlabel("Datum")
ax.set_ylabel("Stopa prinosa")

# Postavljanje xticks svakog četvrtog meseca
ax.set_xticks(ax.get_xticks()[::4])

# Rotiranje natpisa na x-osi radi bolje čitljivosti
plt.xticks(rotation=45)
plt.legend()

st.pyplot(fig)


# Povracaj kroz godine
triangle_option = st.selectbox(
    "Izaberite coin za koji zelite da viidte istorijski povrat!",
    list_of_coins,
    index=list_of_coins.index("BTC"),
    placeholder="Izaberite coin",
)

annual = (
    coins.loc[:, triangle_option]
    .to_frame()
    .dropna()
    .resample("Y", kind="period")
    .last()
)
annual["Return"] = np.log(annual[triangle_option] / annual[triangle_option].shift())
annual = annual.dropna()

years = annual.shape[0]
windows = [year for year in range(years, 0, -1)]

for year in windows:
    # annual["{}Y".format(year)] = annual.Return.rolling(year).mean()
    annual["{}Y".format(year)] = np.exp(year * annual.Return.rolling(year).mean()) * 100

absolute_triangle = annual.drop(columns=[triangle_option, "Return"])

fig, ax = plt.subplots(figsize=(30, 25))
sns.set(font_scale=1.8)
sns.heatmap(
    absolute_triangle,
    annot=True,
    fmt=".0f",
    cmap="RdYlGn",
    vmin=60,
    vmax=5000,
    center=100,
    annot_kws={"size": 20},
)
plt.title("Istorijski prinos na kriptovalutu.")
plt.tick_params(axis="y", labelright=True)
st.pyplot(fig)
