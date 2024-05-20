import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")
pd.options.display.float_format = "{:.4f}".format


# Funckije
def ann_risk_return(returns_df):
    df = returns_df.agg(["mean", "std"]).T
    df.columns = ["Return", "Risk"]
    df.Return = df.Return * 365
    df.Risk = df.Risk * np.sqrt(365)
    return df


def capm_model(returns):
    df = ann_risk_return(returns)
    df["Sharpe"] = (df["Return"].sub(rf[0])) / df["Risk"]
    cov = returns.cov() * 365

    df["SystRisk_var"] = cov.iloc[:, -1]
    df["TotalRisk_var"] = np.power(df.Risk, 2)
    df["UnsystRisk_var"] = df["TotalRisk_var"].sub(df["SystRisk_var"])
    df["beta"] = df.SystRisk_var / df.loc["BTC", "SystRisk_var"]
    df["capm_ret"] = rf[0] + (df.loc["BTC", "Return"] - rf[0]) * df.beta
    df["alpha"] = df.Return - df.capm_ret
    # df["Er"] = rf[0] + df['beta'] * (df.loc["BTC", "Return"])

    df = df[df["beta"] >= 0]

    return df


def lin_reg_plot(df, coin):
    sns.set(font_scale=1.5)
    sns.jointplot(data=df, x="BTC", y=coin, height=10, kind="reg")
    plt.show()


def get_sample_assets(df, num_assets):
    sample_assets = df[df["capm_ret"] <= df["Return"]].sample(num_assets)

    return sample_assets


def lin_reg_plot(df, coin):
    # Postavljanje veličine fonta
    sns.set(font_scale=1.5)

    # Stvaranje regresijskog grafikona
    plt.figure(figsize=(12, 8))
    sns.regplot(
        data=df, x="BTC", y=coin, scatter_kws={"s": 100}, line_kws={"color": "red"}
    )

    # Dodavanje naslova i oznaka osi
    plt.title(f"Linearna regresija izmedju BTC i {coin}", fontsize=18)
    plt.xlabel("BTC", fontsize=15)
    plt.ylabel(coin, fontsize=15)

    # Prikaz grafikona
    plt.show()


def camp_plot(df_sample, df):
    plt.figure(figsize=(15, 8))
    plt.scatter(df_sample.beta, df_sample.Return, alpha=0.7, label="Assets", s=50)
    for i in df_sample.index:
        plt.annotate(
            i,
            xy=(df_sample.loc[i, "beta"] + 0.01, df_sample.loc[i, "Return"] - 0.01),
            size=7,
        )
    plt.plot(df["beta"], df["capm_ret"], color="orange", label="CAPM line")
    plt.scatter(rf[1], rf[0], s=30, marker="o", c="green")
    plt.annotate(
        "Risk Free Asset", xy=(rf[1] + 0.01, rf[0] - 0.01), size=10, color="green"
    )
    plt.xlabel("beta", fontsize=15)
    plt.ylabel("ann. Return", fontsize=15)
    plt.title("CAPM model", fontsize=20)

    # Add legend
    plt.legend()

    plt.show()


def markowitz_chart(df_port, df_sample):
    min_sharpe = df_port["Sharpe"].min()
    max_sharpe = df_port["Sharpe"].max()
    plt.figure(figsize=(15, 9))
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
    plt.scatter(
        df_sample.loc[:, "Risk"],
        df_sample.loc[:, "Return"],
        s=50,
        color="black",
        marker="D",
    )
    plt.xlabel("ann. Risk(std)", fontsize=15)
    plt.ylabel("ann. Return", fontsize=15)
    plt.title("Risk/Return", fontsize=20)
    plt.show()


def optimized_portfolio(df_port, summary):
    min_sharpe = df_port["Sharpe"].min()
    max_sharpe = df_port["Sharpe"].max()
    plt.figure(figsize=(20, 10))
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
        "Max SR Portfolio",
        xy=(summary.loc["MP", "Risk"] - 0.06, summary.loc["MP", "Return"] + 0.01),
        size=20,
        color="black",
    )
    plt.scatter(rf[1], rf[0], s=100, marker="o", c="black")
    plt.annotate(
        "Risk Free Asset", xy=(rf[1] + 0.01, rf[0] - 0.01), size=20, color="black"
    )
    plt.xlabel("ann. Risk(std)", fontsize=20)
    plt.ylabel("ann. Return", fontsize=20)
    plt.tick_params(axis="both", labelsize=15)
    plt.title("The Max Sharpe Ratio Portfolio", fontsize=25)
    plt.plot(
        [rf[1], summary.loc["MP", "Risk"]],
        [rf[0], summary.loc["MP", "Return"]],
        c="black",
    )
    plt.annotate("Capital Market Line", xy=(0.04, 0.16), size=20, color="black")
    plt.show()


###############
# CAPM MODEL
###############

# Load data
# coins = pd.read_csv("../data/coins.csv", index_col=[0], parse_dates=[0])
# coins.columns = [col.replace("-USD", "") for col in coins.columns]
# columns_no_na = coins.iloc[-1].isna()
# columns_no_na = columns_no_na.iloc[columns_no_na.to_numpy().nonzero()].index

# coins = coins.drop(columns=columns_no_na)
# coins = coins.drop(columns=["AUD", "UNI", "VIC", "APE"])

# coins.to_csv('../data/coins_v2.csv')

coins = pd.read_csv("../data/coins_v2.csv", index_col=[0], parse_dates=[0])

# Risk free rate
risk_free_return = 0.0388
risk_free_risk = 0
rf = [risk_free_return, risk_free_risk]

# Return of coins
ret = coins.pct_change()
capm = capm_model(ret)
# capm_sample = get_sample_assets(capm, 5)
capm_sample = capm.loc[["DOGE", "ETH", "BNB", "XRP", "LINK"]]

lin_reg_plot(ret, coin="LINK")
camp_plot(capm_sample, capm)

###############
# PORTFOLIO OPTIMIZATION
###############
port_ret = ret[capm_sample.index].copy()

noa = port_ret.shape[1]
nop = 100000
matrix = np.random.random(noa * nop).reshape(nop, noa)
weights = matrix / matrix.sum(axis=1, keepdims=True)
port_ret_dot = port_ret.dot(weights.T)

port_summary = ann_risk_return(port_ret_dot)
port_summary["Sharpe"] = (port_summary["Return"].sub(rf[0])) / port_summary["Risk"]

markowitz_chart(port_summary, capm_sample)

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

summary = ann_risk_return(port_ret)
summary["Sharpe"] = (summary["Return"].sub(rf[0])) / summary["Risk"]
optimized_portfolio(port_summary, summary)


# def optimized_portfolio():
#     plt.figure(figsize=(20, 10))
#     plt.scatter(
#         port_summary.loc[:, "Risk"],
#         port_summary.loc[:, "Return"],
#         s=20,
#         c=port_summary.loc[:, "Sharpe"],
#         cmap="coolwarm",
#         vmin=0.76,
#         vmax=1.18,
#         alpha=0.8,
#     )
#     plt.colorbar()
#     plt.scatter(
#         summary.loc["MP", "Risk"],
#         summary.loc["MP", "Return"],
#         s=300,
#         c="black",
#         marker="*",
#     )
#     plt.annotate(
#         "Max SR Portfolio",
#         xy=(summary.loc["MP", "Risk"] - 0.06, summary.loc["MP", "Return"] + 0.01),
#         size=20,
#         color="black",
#     )
#     plt.scatter(rf[1], rf[0], s=100, marker="o", c="black")
#     plt.annotate(
#         "Risk Free Asset", xy=(rf[1] + 0.01, rf[0] - 0.01), size=20, color="black"
#     )
#     plt.xlabel("ann. Risk(std)", fontsize=20)
#     plt.ylabel("ann. Return", fontsize=20)
#     plt.tick_params(axis="both", labelsize=15)
#     plt.title("The Max Sharpe Ratio Portfolio", fontsize=25)
#     plt.plot(
#         [rf[1], summary.loc["MP", "Risk"]],
#         [rf[0], summary.loc["MP", "Return"]],
#         c="black",
#     )
#     plt.annotate("Capital Market Line", xy=(0.04, 0.16), size=20, color="black")
#     plt.show()
