"""
使用隐马尔可夫模型 HMM 预测股市状态。

输出重点：
1. 第二日上涨概率 up_probability
2. 第二日预期上涨幅度 expected_up_return_if_up
3. 第二日期望收益 expected_next_return

HMM 不直接预测明天收盘价。它先识别隐藏市场状态，再根据当前状态的转移概率，
推算下一日进入上涨状态的概率和收益特征。
"""

from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd
import tushare as ts
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import config


STOCK_FILE = config.STOCK_FILE
YEARS = 5
N_STATES = 3
MIN_ROWS = 300
MAX_STOCKS = None
REQUEST_SLEEP_SECONDS = 0.2
OUTPUT_FILE = "hmm_market_state_result.csv"


def get_start_date(years=YEARS):
    return (datetime.today() - timedelta(days=365 * years)).strftime("%Y%m%d")


def load_stock_pool(stock_file=STOCK_FILE, max_stocks=MAX_STOCKS):
    stocks = pd.read_csv(stock_file, dtype={"ts_code": str})
    if "ts_code" not in stocks.columns:
        raise ValueError(f"{stock_file} 中没有 ts_code 列")

    columns = ["ts_code"]
    if "name" in stocks.columns:
        columns.append("name")

    stocks = stocks[columns].dropna(subset=["ts_code"]).drop_duplicates("ts_code")
    if max_stocks:
        stocks = stocks.head(max_stocks)
    return stocks.to_dict("records")


def get_daily_data(pro, ts_code, start_date):
    df = pro.daily(ts_code=ts_code, start_date=start_date)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def build_features(df):
    """构造 HMM 观测变量：收益率、日内振幅、成交量变化。"""
    data = df.copy()
    data["return"] = 100 * np.log(data["close"] / data["close"].shift(1))
    data["range"] = 100 * (data["high"] - data["low"]) / data["close"]
    data["volume_change"] = 100 * np.log(data["vol"] / data["vol"].shift(1))
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data


def fit_hmm(feature_df, n_states=N_STATES):
    feature_columns = ["return", "range", "volume_change"]
    scaler = StandardScaler()
    x = scaler.fit_transform(feature_df[feature_columns])

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        random_state=42,
    )
    model.fit(x)
    hidden_states = model.predict(x)
    return model, hidden_states


def describe_states(feature_df, hidden_states):
    """把隐藏状态翻译成收益、波动、成交量特征。"""
    state_df = feature_df.copy()
    state_df["state"] = hidden_states
    total_volatility = state_df["return"].std()

    rows = []
    for state, group in state_df.groupby("state"):
        avg_return = group["return"].mean()
        volatility = group["return"].std()
        avg_range = group["range"].mean()
        avg_volume_change = group["volume_change"].mean()

        if avg_return > 0 and volatility <= total_volatility:
            label = "偏上涨低波动"
        elif avg_return > 0:
            label = "偏上涨高波动"
        elif volatility > total_volatility:
            label = "偏下跌高波动"
        else:
            label = "偏震荡或弱势"

        rows.append(
            {
                "state": int(state),
                "label": label,
                "avg_return": avg_return,
                "volatility": volatility,
                "avg_range": avg_range,
                "avg_volume_change": avg_volume_change,
                "days": len(group),
                "is_up_state": avg_return > 0,
            }
        )

    return pd.DataFrame(rows).sort_values("state").reset_index(drop=True)


def calculate_up_prediction(state_summary, next_probs):
    """计算第二日上涨概率、上涨时预期涨幅、总体期望收益和预期波动。"""
    up_probability = 0.0
    weighted_up_return = 0.0
    expected_next_return = 0.0
    expected_next_volatility = 0.0

    for state, probability in enumerate(next_probs):
        state_info = state_summary[state_summary["state"] == state].iloc[0]
        avg_return = state_info["avg_return"]
        expected_next_return += probability * avg_return
        expected_next_volatility += probability * state_info["volatility"]

        if state_info["is_up_state"]:
            up_probability += probability
            weighted_up_return += probability * avg_return

    expected_up_return_if_up = (
        weighted_up_return / up_probability if up_probability > 0 else 0.0
    )

    return {
        "up_probability": up_probability,
        "expected_up_return_if_up": expected_up_return_if_up,
        "expected_next_return": expected_next_return,
        "expected_next_volatility": expected_next_volatility,
    }


def summarize_prediction(ts_code, name, feature_df, model, hidden_states):
    state_summary = describe_states(feature_df, hidden_states)
    current_state = int(hidden_states[-1])
    next_probs = model.transmat_[current_state]
    up_prediction = calculate_up_prediction(state_summary, next_probs)

    most_likely_next_state = int(np.argmax(next_probs))
    current_label = state_summary.loc[
        state_summary["state"] == current_state, "label"
    ].iloc[0]
    next_label = state_summary.loc[
        state_summary["state"] == most_likely_next_state, "label"
    ].iloc[0]

    row = {
        "ts_code": ts_code,
        "name": name,
        "last_date": feature_df["trade_date"].iloc[-1].strftime("%Y%m%d"),
        "current_state": current_state,
        "current_label": current_label,
        "next_state": most_likely_next_state,
        "next_label": next_label,
        "next_state_probability": next_probs[most_likely_next_state],
        **up_prediction,
    }

    for state, probability in enumerate(next_probs):
        row[f"prob_state_{state}"] = probability

    return row


def analyze_stock(pro, stock, start_date):
    ts_code = stock["ts_code"]
    name = stock.get("name", "")
    df = get_daily_data(pro, ts_code, start_date)
    if len(df) < MIN_ROWS:
        return None

    feature_df = build_features(df)
    if len(feature_df) < MIN_ROWS:
        return None

    model, hidden_states = fit_hmm(feature_df)
    return summarize_prediction(ts_code, name, feature_df, model, hidden_states)


def print_top10(result_df, sort_column, title):
    display_columns = [
        "ts_code",
        "name",
        "current_label",
        "next_label",
        "up_probability",
        "expected_up_return_if_up",
        "expected_next_return",
        "expected_next_volatility",
    ]
    print("\n" + "=" * 80)
    print(title)
    print(
        result_df.sort_values(sort_column, ascending=False)
        .head(10)[display_columns]
        .to_string(
            index=False,
            formatters={
                "up_probability": "{:.2%}".format,
                "expected_up_return_if_up": "{:.4f}%".format,
                "expected_next_return": "{:.4f}%".format,
                "expected_next_volatility": "{:.4f}%".format,
            },
        )
    )


def main():
    ts.set_token(config.TUSHARE_TOKEN)
    pro = ts.pro_api()
    start_date = get_start_date()
    stocks = load_stock_pool()
    results = []

    print(f"开始 HMM 预测，股票数量: {len(stocks)}, 起始日期: {start_date}")

    for index, stock in enumerate(stocks, start=1):
        ts_code = stock["ts_code"]
        try:
            row = analyze_stock(pro, stock, start_date)
            if row is None:
                print(f"[{index}/{len(stocks)}] {ts_code} 数据不足，跳过")
                continue

            results.append(row)
            print(
                f"[{index}/{len(stocks)}] {ts_code} "
                f"上涨概率={row['up_probability']:.2%} "
                f"预期上涨幅度={row['expected_up_return_if_up']:.4f}%"
            )
        except Exception as exc:
            print(f"[{index}/{len(stocks)}] {ts_code} 计算失败: {exc}")

        time.sleep(REQUEST_SLEEP_SECONDS)

    if not results:
        print("没有成功生成 HMM 预测结果")
        return

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print_top10(result_df, "expected_up_return_if_up", "第二日预测涨幅最高的前 10 只")
    print_top10(result_df, "up_probability", "第二日上涨概率最高的前 10 只")
    print(f"\n完整结果已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
