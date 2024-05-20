import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from urllib.error import URLError
import numpy as np

plt.style.use("seaborn-v0_8")
pd.options.display.float_format = "{:.4f}".format


def ann_risk_return(returns_df):
    df = returns_df.agg(["mean", "std"]).T
    df.columns = ["Return", "Risk"]
    df.Return = df.Return * 365
    df.Risk = df.Risk * np.sqrt(365)
    return df


def markowitz_chart(df_port, df_sample):
    min_sharpe = df_port["Sharpe"].min()
    max_sharpe = df_port["Sharpe"].max()
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.scatter(
        df_port.loc[:, "Risk"],
        df_port.loc[:, "Return"],
        s=20,
        c=df_port.loc[:, "Sharpe"],
        cmap="coolwarm",
        vmin=min_sharpe,
        vmax=max_sharpe,
        alpha=0.8,
    )
    ax.scatter(
        df_sample.loc[:, "Risk"],
        df_sample.loc[:, "Return"],
        s=50,
        color="black",
        marker="D",
        label=df_sample.index,
    )
    plt.xlabel("Rizik", fontsize=15)
    plt.ylabel("Prinos", fontsize=15)
    plt.title("Markowitz Chart", fontsize=20)
    st.pyplot(fig)


def optimized_portfolio(df_port, summary):
    min_sharpe = df_port["Sharpe"].min()
    max_sharpe = df_port["Sharpe"].max()
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.scatter(
        df_port.loc[:, "Risk"],
        df_port.loc[:, "Return"],
        s=20,
        c=df_port.loc[:, "Sharpe"],
        cmap="coolwarm",
        vmin=min_sharpe,
        vmax=max_sharpe,
        alpha=0.8,
    )
    plt.colorbar()
    plt.scatter(
        summary.loc["MP", "Risk"],
        summary.loc["MP", "Return"],
        s=300,
        c="black",
        marker="*",
    )
    plt.annotate(
        "Optimalan portfolio!",
        xy=(summary.loc["MP", "Risk"] - 0.06, summary.loc["MP", "Return"] + 0.01),
        size=20,
        color="black",
    )
    plt.scatter(
        summary.loc[:, "Risk"],
        summary.loc[:, "Return"],
        s=50,
        color="black",
        marker="D",
        label=summary.index,
    )
    plt.scatter(rf[1], rf[0], s=100, marker="o", c="black")
    plt.annotate(
        "Stopa bez rizika", xy=(rf[1] + 0.01, rf[0] - 0.01), size=20, color="black"
    )
    plt.xlabel("Rizik", fontsize=20)
    plt.ylabel("Prinos", fontsize=20)
    plt.tick_params(axis="both", labelsize=15)
    plt.title("Optimizacija portfolija", fontsize=25)
    plt.plot(
        [rf[1], summary.loc["MP", "Risk"]],
        [rf[0], summary.loc["MP", "Return"]],
        c="black",
    )
    # plt.annotate("Capital Market Line", xy=(0.04, 0.16), size=20, color="black")
    st.pyplot(fig)


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


st.markdown("# Optimizacija Portfolija")

risk_free_return = 0.0388
risk_free_risk = 0
rf = [risk_free_return, risk_free_risk]

coins = get_all_coins()
list_of_coins = list(coins.columns)

capm = get_capm_data()
# st.dataframe(capm)

ret = get_return_data()
# st.dataframe(ret)

crypto_option = st.multiselect("Koju kriptovalutu zelite da analizirate?", capm.index)
# st.write(crypto_option)

if st.button("Izracunaj!", type="primary"):
    st.write("Sacekajte da se izracuna.")
    # if len(crypto_option) >= 5:
    # port_ret = capm.loc[crypto_option]
    port_ret = ret[crypto_option].copy().dropna()
    # st.dataframe(port_ret)

    noa = port_ret.shape[1]
    nop = 100000
    matrix = np.random.random(noa * nop).reshape(nop, noa)
    weights = matrix / matrix.sum(axis=1, keepdims=True)
    port_ret_dot = port_ret.dot(weights.T)

    port_summary = ann_risk_return(port_ret_dot)
    port_summary["Sharpe"] = (port_summary["Return"].sub(rf[0])) / port_summary["Risk"]
    # st.dataframe(port_summary)

    # # print(port_ret_dot)
    # st.dataframe(capm.loc[crypto_option])
    # markowitz_chart(port_summary, capm.loc[crypto_option])

    # Optimizaciona tangenta
    import scipy.optimize as sco  # import scipy optimize

    def port_ret_func(eweights):
        return port_ret.dot(eweights.T).mean() * 365

    def port_vol(eweights):
        return port_ret.dot(eweights.T).std() * np.sqrt(365)

    def min_func_sharpe(eweights):
        return (rf[0] - port_ret_func(eweights)) / port_vol(eweights)

    eweights = np.full(noa, 1 / noa)

    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bnds = tuple((0, 1) for x in range(noa))

    opts = sco.minimize(
        min_func_sharpe, eweights, method="SLSQP", bounds=bnds, constraints=cons
    )
    optimal_weights = opts["x"]
    opt_weights = pd.Series(index=port_ret.columns, data=optimal_weights)

    port_ret["MP"] = port_ret.dot(opt_weights)
    # st.dataframe(port_ret)

    summary = ann_risk_return(port_ret)
    summary["Sharpe"] = (summary["Return"].sub(rf[0])) / summary["Risk"]
    optimized_portfolio(port_summary, summary)

    opt_weights["MP"] = 1
    opt_weights = np.round(opt_weights * 100, 2)
    opt_weights = opt_weights.to_frame()
    opt_weights.columns = ["% portfolio"]
    summary_final = pd.merge(
        left=summary, right=opt_weights, left_index=True, right_index=True
    )

    st.dataframe(summary_final)
    # st.dataframe(opt_weights)
