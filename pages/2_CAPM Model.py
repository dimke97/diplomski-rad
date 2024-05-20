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


def ann_risk_return(returns_df):
    df = returns_df.agg(["mean", "std"]).T
    df.columns = ["Return", "Risk"]
    df.Return = df.Return * 365
    df.Risk = df.Risk * np.sqrt(365)
    return df


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
list_of_coins = list(coins.columns)

capm = get_capm_data()
ret = get_return_data()


crypto_option = st.selectbox(
    "Izaberite coin za koji zelite da vidite odnos sa BTC!",
    list_of_coins,
    index=list_of_coins.index("ETH"),
    placeholder="Izaberite coin",
)

st.write(f"Regresija izmedju BTC i {crypto_option}")

fig, ax = plt.subplots()
s = sns.lmplot(
    data=ret[["BTC", crypto_option]].dropna(),
    x="BTC",
    y=crypto_option,
    height=10,
    line_kws={"color": "red"},
)
plt.title(f"Korelacija izmedju BTC i {crypto_option}")
plt.legend()
st.pyplot(s)

st.write(
    "Izaberite 5 kriptovaluta ili pritisnite dugme da vam algoritam sam da preporuku."
)


def show_sec_line(capm_sample):
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.lineplot(
        capm,
        x="beta",
        y="capm_ret",
        label="Security Market Line",
    )
    sns.scatterplot(
        data=capm_sample,
        x=capm_sample.beta,
        y=capm_sample.Return,
        # hue=capm_sample.Sharpe,
        s=200,
    )
    for i in capm_sample.index:
        plt.annotate(
            i,
            xy=(
                capm_sample.loc[i, "beta"] + 0.02,
                capm_sample.loc[i, "Return"] - 0.02,
            ),
            size=14,
        )
    # plt.xlim(0, 1000)  # Ograničava x os od 20 do 80
    # plt.ylim(0, 1000)  # Ograničava y os od 0 do 5000
    plt.title("CAPM model")
    plt.xlabel("Beta")
    plt.ylabel("Prinos")
    # plt.legend()
    st.pyplot(fig)


radio_choice_random = st.radio(
    "Da li zelite da Vam preporucimo kriptovalute ili ćete ih sami izabrati?",
    [
        "Želim sam da izaberem kriptovalute!",
        "Želim da pogledam vaše predloge!",
    ],
    captions=[
        "Izaberite valute iz padajućeg menija.",
        "Izaberite koliki želite da Vam bude portfolio.",
    ],
)

if radio_choice_random == "Želim sam da izaberem kriptovalute!":
    st.write("Samostalno.")
    crypto_option = st.multiselect(
        "Koju kriptovalutu zelite da analizirate?", capm.index
    )
    cypto_options_df = capm.loc[crypto_option]
    st.dataframe(cypto_options_df)
    show_sec_line(cypto_options_df)

    # if st.button("Sacuvaj!", type="primary"):
    #     capm.loc[crypto_option].to
else:
    st.write("Generiši mi portfolio")
    number_of_coins = st.number_input(
        "Insert a number",
        min_value=1,
        max_value=20,
        value=None,
        placeholder="Izaberite broj kriptovaluta...",
        format="%i",
    )
    capm_recom = capm[capm.Return >= capm.Er]
    if number_of_coins is not None:
        cypto_options_df = capm_recom.sample(number_of_coins)
        st.dataframe(cypto_options_df)
        show_sec_line(cypto_options_df)


# st.markdown("# Optimizacija Portfolija")
# # crypto_option = st.multiselect("Koju kriptovalutu zelite da analizirate?", capm.index)
# # st.write(crypto_option)
# crypto_option = cypto_options_df.index.to_list()
# if st.button("Izracunaj!", type="primary"):
#     st.write("Sacekajte da se izracuna.")
#     # if len(crypto_option) >= 5:
#     # port_ret = capm.loc[crypto_option]
#     port_ret = ret[crypto_option].copy().dropna()
#     # st.dataframe(port_ret)

#     noa = port_ret.shape[1]
#     nop = 100000
#     matrix = np.random.random(noa * nop).reshape(nop, noa)
#     weights = matrix / matrix.sum(axis=1, keepdims=True)
#     port_ret_dot = port_ret.dot(weights.T)

#     port_summary = ann_risk_return(port_ret_dot)
#     port_summary["Sharpe"] = (port_summary["Return"].sub(rf[0])) / port_summary["Risk"]
#     # st.dataframe(port_summary)

#     # # print(port_ret_dot)
#     # st.dataframe(capm.loc[crypto_option])
#     # markowitz_chart(port_summary, capm.loc[crypto_option])

#     # Optimizaciona tangenta
#     import scipy.optimize as sco  # import scipy optimize

#     def port_ret_func(eweights):
#         return port_ret.dot(eweights.T).mean() * 365

#     def port_vol(eweights):
#         return port_ret.dot(eweights.T).std() * np.sqrt(365)

#     def min_func_sharpe(eweights):
#         return (rf[0] - port_ret_func(eweights)) / port_vol(eweights)

#     eweights = np.full(noa, 1 / noa)

#     cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
#     bnds = tuple((0, 1) for x in range(noa))
