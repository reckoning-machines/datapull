from typing import Dict, Tuple
from textblob import TextBlob
import pandas_ta as pta
from pandas.tseries.offsets import MonthEnd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import pandas as pd
import numpy as np


def preprocess_earning_calendar(
    earning_calendar: pd.DataFrame,
) -> pd.DataFrame:
    earning_calendar["fiscal_date_ending"] = pd.to_datetime(
        earning_calendar["fiscal_date_ending"], format="%Y-%m-%d"
    ) + MonthEnd(0)
    earning_calendar = (
        earning_calendar.groupby(["symbol", "date"]).first().reset_index()
    )

    return earning_calendar.astype(str)


def preprocess_fred_data(fred_data: pd.DataFrame) -> pd.DataFrame:
    fred_data["date"] = pd.to_datetime(fred_data["date"]).dt.date
    fred_data["date"] = pd.to_datetime(fred_data["date"])
    fred_data = fred_data.ffill()
    fred_data = fred_data.drop(columns="end_date")
    fred_data["month"] = fred_data["date"].dt.month.astype(str)
    fred_data["day"] = fred_data["date"].dt.day.astype(str)
    return fred_data


def preprocess_etf_prices(
    etf_prices: pd.DataFrame,
) -> pd.DataFrame:
    etf_prices = etf_prices.drop(columns=["gdef", "gvip"])
    return etf_prices


def preprocess_historical_price_full(
    historical_price_full: pd.DataFrame,
) -> pd.DataFrame:
    # target vector created here
    historical_price_full = historical_price_full.sort_values("date", ascending=False)
    historical_price_full = historical_price_full.loc[
        historical_price_full["date"] > "2015/01/01"
    ]

    df_list = []

    for k, g in historical_price_full.groupby("symbol"):
        g = g.sort_values("date", ascending=True)
        g["rsi"] = pta.rsi(g["close"], length=14)
        g["symbol"] = k
        g = g.dropna()
        df_list.append(g)
    historical_price_full = pd.concat(df_list)

    return historical_price_full


def sentiment_score(txt):
    res = TextBlob(txt)
    return res.sentiment.polarity


def preprocess_analyst_estimates(analyst_estimates: pd.DataFrame) -> pd.DataFrame:
    return analyst_estimates


def preprocess_earnings_calls(
    earnings_calls_2020: pd.DataFrame,
    earnings_calls_2021: pd.DataFrame,
    earnings_calls_2022: pd.DataFrame,
    earnings_calls_2023: pd.DataFrame,
    earnings_calls_2024: pd.DataFrame,
) -> pd.DataFrame:
    earnings_calls_2020["sentiment_score"] = earnings_calls_2020.apply(
        lambda row: sentiment_score(row["content"]), axis=1
    )
    earnings_calls_2021["sentiment_score"] = earnings_calls_2021.apply(
        lambda row: sentiment_score(row["content"]), axis=1
    )
    earnings_calls_2022["sentiment_score"] = earnings_calls_2022.apply(
        lambda row: sentiment_score(row["content"]), axis=1
    )
    earnings_calls_2023["sentiment_score"] = earnings_calls_2023.apply(
        lambda row: sentiment_score(row["content"]), axis=1
    )
    earnings_calls_2024["sentiment_score"] = earnings_calls_2024.apply(
        lambda row: sentiment_score(row["content"]), axis=1
    )
    earnings_calls = pd.concat(
        [
            earnings_calls_2020,
            earnings_calls_2021,
            earnings_calls_2022,
            earnings_calls_2023,
            earnings_calls_2024,
        ]
    )
    df_list = []
    for k, g in earnings_calls.groupby("symbol"):
        print(k)
        for i, row in g.iterrows():
            txt = row["content"]
            symbol = row["symbol"]
            quarter = row["quarter"]
            year = row["year"]
            date = row["date"]

            txt_list = txt.splitlines()
            for txt_item in txt_list:
                try:
                    speaker, stmt = txt_item.split(":")[0:2]
                except:
                    stmt = txt_item
                    print(f"split error: {txt_item}")
                    # continue
                stmt_list = stmt.split(".")
                for stmt_item in stmt_list:
                    s = sentiment_score(stmt_item)
                    blob = TextBlob(stmt_item)
                    noun_phrases = blob.noun_phrases
                    noun_phrases = [x for x in noun_phrases if "’" not in x]
                    data = [
                        [
                            symbol,
                            quarter,
                            year,
                            date,
                            stmt_item,
                            s,
                            speaker,
                            noun_phrases,
                        ]
                    ]
                    df = pd.DataFrame(
                        data,
                        columns=[
                            "symbol",
                            "quarter",
                            "year",
                            "date",
                            "statement",
                            "sentiment",
                            "speaker",
                            "noun_phrases",
                        ],
                    )
                    # print(df)
                    df_list.append(df)
    df = pd.concat(df_list)
    df = df.loc[df["statement"] != ""]
    df = df.loc[df["speaker"] != "Operator"]
    df = df.loc[df["sentiment"] != 0]
    df = df.loc[df["statement"].str.strip() != "Okay"]
    # pd.set_option("display.max_colwidth", None)

    return earnings_calls, df


def betas_table_model_input(
    preprocessed_fred_data: pd.DataFrame,
    preprocessed_historical_price_full: pd.DataFrame,
    preprocessed_etf_prices: pd.DataFrame,
    preprocessed_earning_calendar: pd.DataFrame,
) -> pd.DataFrame:
    # exog_list = exog["symbol"].to_list()
    key_cols = [
        "close",
        "earnings_close",
        "hyg",
        "tlt",
        "vb",
        "vtv",
        "vug",
        "rut",
        "spx",
        "DGS10",
        "DGS2",
        "DTB3",
        "DFF",
        "T10Y2Y",
        "T5YIE",
        "BAMLH0A0HYM2",
        "DEXUSEU",
        "KCFSI",
        "DRTSCILM",
        "RSXFS",
        "MARTSMPCSM44000USS",
        "H8B1058NCBCMG",
        "DCOILWTICO",
        "VXVCLS",
        "H8B1247NCBCMG",
        "SP500",
        "GASREGW",
        "CSUSHPINSA",
        "UNEMPLOY",
        # "spx",
    ]
    t_plus = 1

    symbol_list = preprocessed_historical_price_full["symbol"].to_list()
    # symbol_list = ["PNC"]
    #    symbol_list = [x + " US" for x in symbol_list]

    preprocessed_historical_price_full = preprocessed_historical_price_full.loc[
        preprocessed_historical_price_full["symbol"].isin(symbol_list)
    ]

    preprocessed_earning_calendar.loc[
        preprocessed_earning_calendar["symbol"].isin(symbol_list)
    ]

    preprocessed_earning_calendar = preprocessed_earning_calendar.sort_values("date")
    # preprocessed_earning_calendar = (
    #    preprocessed_earning_calendar.groupby("date").last().reset_index()
    # )

    preprocessed_historical_price_full["date"] = pd.to_datetime(
        preprocessed_historical_price_full["date"]
    )
    preprocessed_etf_prices["date"] = pd.to_datetime(preprocessed_etf_prices["date"])
    preprocessed_fred_data["date"] = pd.to_datetime(preprocessed_fred_data["date"])

    df_list = []
    for k, g in preprocessed_earning_calendar.groupby("symbol"):
        c = preprocessed_earning_calendar.loc[
            preprocessed_earning_calendar["symbol"] == k
        ]
        c["date"] = pd.to_datetime(c["date"])
        g["date"] = pd.to_datetime(g["date"])
        g = g.resample("D", on="date").mean(numeric_only=True).reset_index()[["date"]]
        # g = g[g["date"] < "2024-03-01"]
        g = g.merge(c, how="left", left_on=["date"], right_on=["date"])
        g["fiscal_date_ending"] = g["fiscal_date_ending"].bfill()
        g["symbol"] = g["symbol"].bfill()
        g = g.loc[~g["fiscal_date_ending"].isna()]
        df_list.append(g)

    c = pd.concat(df_list)
    p = preprocessed_historical_price_full

    p["date"] = pd.to_datetime(p["date"])
    p_merge = c.merge(
        p, how="left", left_on=["symbol", "date"], right_on=["symbol", "date"]
    )
    p_merge = p_merge.loc[~p_merge["close"].isna()][
        [
            "date",
            "symbol",
            "eps",
            "eps_estimated",
            "fiscal_date_ending",
            "close",
            "rsi",
            # "pct_bbu",
            # "pct_bbl",
            # "MACD_12_26_9",
            "time",
        ]
    ]

    p_merge["previous_date"] = p_merge["date"].shift(1)
    p_merge["previous_close"] = p_merge["close"].shift(1)

    p_merge["earnings_close"] = np.where(
        p_merge["time"] == "bmo", p_merge["previous_close"], p_merge["close"]
    )
    p_merge["earnings_close_date"] = np.where(
        p_merge["time"] == "bmo", p_merge["previous_date"], p_merge["date"]
    )

    p_merge["earnings_trade_date"] = np.where(p_merge["eps_estimated"].isna(), 0, 1)

    # p_merge = p_merge.loc[p_merge["symbol"] == "PNC"]

    def rolling_mean(x):
        x["daily_average_return"] = x["close_return"].rolling(60).mean()
        return x

    def returns(x):
        for col in key_cols:
            x[col + "_return"] = x[col].pct_change()
        return x

    def one_day_returns(x):
        key_cols = ["earnings_close", "close", "spx"]
        # key_cols = ["earnings_close", "close", "xlf"]
        for col in key_cols:
            x[col + "_one_day_return"] = x[col].pct_change(t_plus)
        return x

    def thirty_day_returns(x):
        key_cols = ["close", "spx"]
        # key_cols = ["close", "xlf"]
        for col in key_cols:
            x[col + "_thirty_day_return"] = x[col].pct_change(30)
        return x

    # p_merge = p_merge.loc[p_merge["symbol"] == "PNC"]
    # print(p_merge.tail())

    p_merge = p_merge.merge(
        preprocessed_etf_prices,
        how="inner",
        left_on=["date"],
        right_on=["date"],
    )

    p_merge = p_merge.merge(
        preprocessed_fred_data, how="left", left_on=["date"], right_on=["date"]
    )

    p_merge = p_merge.drop(columns=["symbol_y"])
    p_merge = p_merge.drop(columns=["symbol"])

    p_merge = p_merge.rename(columns={"symbol_x": "symbol"})

    # p_merge = p_merge.loc[~p_merge["month"].isna()]

    p_merge = p_merge.sort_values(["symbol", "date"], ascending=True)

    p_merge = p_merge.groupby(["symbol"]).apply(returns)
    p_merge = p_merge.drop(columns=["symbol"])
    p_merge = p_merge.reset_index()

    p_merge = p_merge.groupby(["symbol"]).apply(rolling_mean)
    p_merge = p_merge.drop(columns=["level_1"])
    p_merge = p_merge.drop(columns=["symbol"])

    p_merge = p_merge.reset_index()
    p_merge = p_merge.drop(columns=["level_1"])

    p_merge = p_merge.groupby(["symbol"]).apply(one_day_returns)
    p_merge = p_merge.drop(columns=["symbol"])
    p_merge = p_merge.reset_index()
    p_merge = p_merge.drop(columns=["level_1"])
    p_merge = p_merge.groupby(["symbol"]).apply(thirty_day_returns)
    p_merge = p_merge.drop(columns=["symbol"])
    p_merge = p_merge.reset_index()
    p_merge = p_merge.drop(columns=["level_1"])

    # p_merge["close_one_day_return"] = p_merge["close_one_day_return"].shift(-1)

    p_merge["earnings_close_one_day_return"] = p_merge[
        "earnings_close_one_day_return"
    ].shift(-1)

    # p_merge["spx_one_day_return"] = p_merge["spx_one_day_return"].shift(-1)
    p_merge["close_thirty_day_return"] = p_merge["close_thirty_day_return"].shift(-30)
    p_merge["spx_thirty_day_return"] = p_merge["spx_thirty_day_return"].shift(-30)
    # p_merge["spx_thirty_day_return"] = p_merge["spx_thirty_day_return"].shift(-30)

    #    print(p_merge.head())
    #    quit()
    #    p_merge = p_merge.drop(columns=["symbol"])
    #    p_merge = p_merge.reset_index()
    #    p_merge = p_merge.drop(columns=["level_1"])
    betas_table = create_betas_table(p_merge, key_cols)

    return betas_table


def create_betas_table(p_merge: pd.DataFrame, key_cols) -> pd.DataFrame:
    # exog_list = exog["symbol"].to_list()

    beta_window = 504

    return_cols = []
    for col in key_cols:
        col = col + "_return"
        return_cols.append(col)

    return_cols.append("close_return")

    # add log returns to historical prices
    window = beta_window
    df_out = []
    return_cols = list(set(return_cols))
    for k, g in p_merge.groupby("symbol"):
        g[return_cols] = g[return_cols].replace(-np.inf, np.nan)
        g[return_cols] = g[return_cols].replace(np.inf, np.nan)
        g[return_cols] = g[return_cols].fillna(0)

        if len(g) < window:
            window = len(g)
        for col in return_cols:
            if col != "close_return":
                model = RollingOLS.from_formula(
                    f"close_return ~ {col}", data=g[return_cols], window=window
                )
                rres = model.fit()
                g[col + "_coef"] = rres.params[col].values
        df_out.append(g)

    p_merge = pd.concat(df_out)
    p_merge = p_merge.sort_values(["symbol", "date"])
    p_merge["symbol"] = p_merge["symbol"] + " US"
    p_merge["date"] = pd.to_datetime(p_merge["date"]).dt.date
    p_merge = p_merge.sort_values(["symbol", "date"])
    p_merge = p_merge.ffill()

    # p_merge["spx_return_coef"] = p_merge["spx_beta"]

    p_merge["datapull_date"] = p_merge["datapull_date_y"]

    p_merge.drop(columns=["datapull_date_x", "datapull_date_y"], inplace=True)
    # print(p_merge.tail())

    # p_merge["one_day_alpha"] = (
    #     p_merge["close_one_day_return"]
    #     - p_merge["spx_return_coef"] * p_merge["spx_one_day_return"]
    # )
    p_merge["one_day_alpha"] = (
        p_merge["close_one_day_return"]
        - p_merge["spx_return_coef"] * p_merge["spx_one_day_return"]
    )

    # print(
    #     p_merge[
    #         [
    #             "symbol",
    #             "date",
    #             "close_one_day_return",
    #             "spx_one_day_return",
    #             "spx_return_coef",
    #             "one_day_alpha",
    #         ]
    #     ].tail(40)
    # )
    # quit()

    p_merge["thirty_day_alpha"] = (
        p_merge["close_thirty_day_return"]
        - p_merge["spx_return_coef"] * p_merge["spx_thirty_day_return"]
    )

    p_merge["earnings_one_day_alpha"] = (
        p_merge["earnings_close_one_day_return"]
        - p_merge["spx_return_coef"] * p_merge["spx_one_day_return"]
    )

    # p_merge["thirty_day_alpha"] = (
    #     p_merge["close_thirty_day_return"]
    #     - p_merge["spx_return_coef"] * p_merge["spx_thirty_day_return"]
    # )

    # p_merge["earnings_one_day_alpha"] = (
    #     p_merge["earnings_close_one_day_return"]
    #     - p_merge["spx_return_coef"] * p_merge["spx_one_day_return"]
    # )

    keep_cols = [
        "symbol",
        "date",
        "close",
        "rsi",
        # "pct_bbu",
        # "pct_bbl",
        # "MACD_12_26_9",
        "one_day_alpha",
        "thirty_day_alpha",
        "earnings_one_day_alpha",
        "earnings_close_date",
        "earnings_trade_date",
        "fiscal_date_ending",
        "daily_average_return",
    ]
    for col in p_merge.columns:
        if "_return" in col:
            keep_cols.append(col)

    betas_table = p_merge[keep_cols]

    # betas_table.columns = [
    #     "symbol",
    #     "date",
    #     "close",
    #     "rsi",
    #     "close_one_day_return",
    #     "close_thirty_day_return",
    #     "close_beta_spx_return",
    #     "small_cap_factor",
    #     "value_factor",
    #     "growth_factor",
    #     "high_yield_factor",
    #     # "defensive_factor",
    #     # "hedgefund_vip_factor",
    #     "treasuries_price_factor",
    #     "spx_one_day_return",
    #     "spx_thirty_day_return",
    #     "one_day_alpha",
    #     "thirty_day_alpha",
    #     "earnings_trade_date",
    #     "fiscal_date_ending",
    # ]

    betas_table["alpha_target"] = np.where(betas_table["thirty_day_alpha"] > 0, 1, 0)

    # betas_table = betas_table.loc[~betas_table["one_day_alpha"].isna()]
    # print(betas_table["date"].max())
    #    betas_table = betas_table[
    #        ["symbol", "date", "close", "close_one_day_return", "spx_one_day_return"]
    #    ]
    #    print(betas_table.tail(40))
    #    quit()

    return betas_table
