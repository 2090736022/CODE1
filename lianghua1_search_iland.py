import time
from pathlib import Path

import pandas as pd

import config
from lianghua1_search import (
    create_pro as create_main_pro,
    ensure_cache_dirs as ensure_shared_cache_dirs,
    get_daily_basic as get_shared_daily_basic,
    get_daily_data as get_shared_daily_data,
)
from progress import show_progress
from stock_filters import filter_non_st_stocks


TOKEN = config.TUSHARE_TOKEN
MIN_TOTAL_MV = config.MIN_TOTAL_MV
BASE_DIR = Path(__file__).resolve().parent
SEARCH_CACHE_DIR = BASE_DIR / "model_artifacts" / "download_daily"
DAILY_CACHE_DIR = SEARCH_CACHE_DIR / "daily"
DAILY_BASIC_CACHE_DIR = SEARCH_CACHE_DIR / "daily_basic"
DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
DAILY_BASIC_FIELDS = "ts_code,trade_date,turnover_rate,total_mv"
DOWNLOAD_YEARS = 5
BACKTEST_FORWARD_DAYS = config.BACKTEST_FORWARD_DAYS


def log(message):
    """统一打印主程序运行进度。"""
    print(f"[main] {message}")


def create_pro():
    """创建 Tushare pro 接口对象。"""
    return create_main_pro()


def ensure_cache_dirs():
    ensure_shared_cache_dirs()


def cache_code(ts_code):
    return str(ts_code).replace(".", "_")


def five_year_start_date(end_date):
    return (pd.to_datetime(str(end_date)) - pd.DateOffset(years=DOWNLOAD_YEARS)).strftime("%Y%m%d")


def read_cached_csv(cache_file):
    cache_file = Path(cache_file)
    if not cache_file.exists():
        return None
    try:
        return pd.read_csv(cache_file, dtype={"ts_code": str, "trade_date": str})
    except Exception as error:
        print(f"{cache_file} 读取失败，将重新下载: {error}")
        return None


def get_daily_data(pro, ts_code, start_date, end_date):
    """优先读取与主筛选脚本一致的本地日线缓存。"""
    return get_shared_daily_data(pro, ts_code, start_date, end_date)


def normalize_daily_frame(df):
    if df is None or df.empty:
        return df

    data = df.copy()
    for column in ("open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "trade_date" in data.columns:
        data["trade_date"] = data["trade_date"].astype(str)
    return data


def calculate_forward_5d_return(ts_code, trade_date):
    cache_file = DAILY_CACHE_DIR / f"{cache_code(ts_code)}.csv"
    daily_df = read_cached_csv(cache_file)
    if daily_df is None or daily_df.empty:
        return None

    daily_df = normalize_daily_frame(daily_df)
    daily_df = daily_df.drop_duplicates(subset=["trade_date"], keep="last")
    daily_df = daily_df.sort_values("trade_date").reset_index(drop=True)
    daily_df["trade_date"] = daily_df["trade_date"].astype(str)

    matched_index = daily_df.index[daily_df["trade_date"] == str(trade_date)]
    if len(matched_index) == 0:
        return None

    selected_index = matched_index[0]
    future_df = daily_df.iloc[selected_index + 1 : selected_index + 1 + BACKTEST_FORWARD_DAYS].copy()
    if len(future_df) < BACKTEST_FORWARD_DAYS:
        return None

    future_df["open"] = pd.to_numeric(future_df["open"], errors="coerce")
    future_df["close"] = pd.to_numeric(future_df["close"], errors="coerce")
    future_df = future_df.dropna(subset=["open", "close"]).reset_index(drop=True)
    if len(future_df) < 2:
        return None

    best_result = None
    for open_index in range(len(future_df) - 1):
        open_row = future_df.iloc[open_index]
        open_price = open_row["open"]
        if pd.isna(open_price) or open_price <= 0:
            continue

        later_df = future_df.iloc[open_index + 1 :]
        if later_df.empty:
            continue

        close_index = later_df["close"].idxmax()
        close_row = future_df.loc[close_index]
        close_price = close_row["close"]
        if pd.isna(close_price):
            continue

        return_ratio = close_price / open_price
        if best_result is None or return_ratio > best_result["future_5d_high_close_to_low_open_return"]:
            best_result = {
                "future_5d_lowest_open": open_price,
                "future_5d_lowest_open_date": open_row["trade_date"],
                "future_5d_highest_close": close_price,
                "future_5d_highest_close_date": close_row["trade_date"],
                "future_5d_high_close_to_low_open_return": return_ratio,
            }

    if best_result is None:
        return None

    return {
        **best_result,
    }


def attach_backtest_result(result_df):
    backtest_columns = [
        "future_5d_lowest_open",
        "future_5d_lowest_open_date",
        "future_5d_highest_close",
        "future_5d_highest_close_date",
        "future_5d_high_close_to_low_open_return",
    ]
    if result_df is None or result_df.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date"] + backtest_columns)

    rows = []
    for _, row in result_df.iterrows():
        backtest = calculate_forward_5d_return(row["ts_code"], row["trade_date"])
        if backtest is None:
            continue

        output_row = row.to_dict()
        output_row.update(backtest)
        rows.append(output_row)

    if not rows:
        return pd.DataFrame(columns=list(result_df.columns) + backtest_columns)

    return pd.DataFrame(rows)


def is_limit_up(row):
    """判断单日是否涨停，优先用 pct_chg，缺失时用收盘价/昨收价估算。"""
    pct_chg = pd.to_numeric(row.get("pct_chg"), errors="coerce")
    if pd.notna(pct_chg) and pct_chg >= config.LIMIT_UP_PCT:
        return True

    close = pd.to_numeric(row.get("close"), errors="coerce")
    pre_close = pd.to_numeric(row.get("pre_close"), errors="coerce")
    if pd.isna(close) or pd.isna(pre_close) or pre_close <= 0:
        return False

    return close >= pre_close * (1 + config.LIMIT_UP_PCT / 100)


def add_macd(df, fast=12, slow=26, signal=9):
    close = pd.to_numeric(df["close"], errors="coerce")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    df["diff"] = ema_fast - ema_slow
    df["dea"] = df["diff"].ewm(span=signal, adjust=False).mean()
    df["macd"] = (df["diff"] - df["dea"]) * 2
    #print(df["macd"])
    return df


def ma60_ma120_upward(df):
    close = pd.to_numeric(df["close"], errors="coerce")
    if len(close) < 121 or close.tail(121).isna().any():
        return False

    ma60_today = close.tail(60).mean()
    ma60_prev = close.iloc[-61:-1].mean()
    ma120_today = close.tail(120).mean()
    ma120_prev = close.iloc[-121:-1].mean()
    return (
        ma60_today > ma60_prev
        and ma120_today > ma120_prev
        and ma60_today >= ma120_today * 1.01
    )


def volume_above_prev_20d_avg(df, ratio=1.05):
    vol = pd.to_numeric(df["vol"], errors="coerce")
    if len(vol) < 21 or vol.tail(21).isna().any():
        return False

    prev_20d_avg_vol = vol.iloc[-21:-1].mean()
    return vol.iloc[-1] > prev_20d_avg_vol * ratio


def has_limit_up_history_in_prev_months(df, months=12):
    data = normalize_daily_frame(df).sort_values("trade_date").reset_index(drop=True)
    if len(data) < 3:
        return False

    trade_dates = pd.to_datetime(data["trade_date"], errors="coerce")
    selected_date = trade_dates.iloc[-1]
    if pd.isna(selected_date):
        return False

    start_date = selected_date - pd.DateOffset(months=months)
    window_df = data[(trade_dates >= start_date) & (trade_dates < selected_date)].copy()
    if len(window_df) < 3:
        return False

    limit_up_flags = window_df.apply(is_limit_up, axis=1).astype(int).reset_index(drop=True)
    if limit_up_flags.rolling(5).sum().ge(2).any() or limit_up_flags.sum() >= 5:
        return True

    return False


def match_strategy(df):
    """日线筛选条件：近三日没涨停、涨幅超 7%、放量 2 倍、站上 MA20/MA120。"""
    min_rows = max(121, config.MIN_DAILY_ROWS_FOR_SEARCH)
    if df is None or len(df) < min_rows:
        return False

    df = df.sort_values("trade_date").reset_index(drop=True).copy()

    numeric_columns = ["open", "high", "close", "pre_close", "pct_chg", "vol"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[numeric_columns].tail(120).isna().any().any():
        return False

    df = add_macd(df)
    if df[["diff", "dea", "macd"]].tail(3).isna().any().any():
        return False

    last_day = df.iloc[-1]
    prev_day = df.iloc[-2]
    prev_prev_day = df.iloc[-3]
    island_reversal = (
        prev_day["vol"] < prev_prev_day["vol"]
        and prev_day["open"] < prev_prev_day["close"]
        and prev_day["pct_chg"] < 0
        and prev_day["close"] > prev_day["open"]
        and last_day["open"] > prev_day["close"]
        and last_day["vol"] > prev_day["vol"]
        and last_day["close"] > last_day["open"]
        and last_day["pct_chg"] > 5
    )
    volume_above_20d_avg = volume_above_prev_20d_avg(df)
    has_prev_year_limit_up_history = has_limit_up_history_in_prev_months(df)
    #if last_day["pct_chg"]>9.8: volume_double_yesterday=True
    ma20 = df["close"].tail(21).mean()
    ma120 = df["close"].tail(120).mean()
    ma5 = df["close"].tail(5).mean()
    ma10 = df["close"].tail(10).mean()
    close_above_ma20_ma120 = last_day["close"] > ma20 and last_day["close"] > ma120
    high_above_ma5_ma10 = last_day["high"] > ma5 and last_day["high"] > ma10
    ##close_above_ma20_ma120 =last_day["close"] > ma120
    moving_average_upward = ma60_ma120_upward(df)
    macd_condition = (
        last_day["macd"] < 0
        and last_day["macd"] > prev_day["macd"]
        and prev_day["macd"] < prev_prev_day["macd"]
        and last_day["diff"] > 0
        and last_day["dea"] > 0
    )
    #print(island_reversal,volume_above_20d_avg,close_above_ma20_ma120,macd_condition,high_above_ma5_ma10,moving_average_upward)
    return (
        island_reversal
        and volume_above_20d_avg
        and has_prev_year_limit_up_history
        and close_above_ma20_ma120
        and high_above_ma5_ma10
        and moving_average_upward
        and macd_condition
    )


def build_pool_from_search_results(result_dir):
    """汇总筛选结果，生成去重后的 result.csv 股票池。"""
    result_df = result_dir
    result_dir = Path(config.RESULT_DIR)
    output_file = result_dir / "result.csv"
    seen_codes = set()
    result_rows = []

    result_dir.mkdir(exist_ok=True)

    if output_file.exists():
        try:
            old_result_df = pd.read_csv(output_file)
        except Exception as e:
            print(f"{output_file} 读取失败: {e}")
            old_result_df = pd.DataFrame(columns=["ts_code", "trade_date"])

        if "ts_code" in old_result_df.columns:
            for _, row in old_result_df.iterrows():
                ts_code = str(row.get("ts_code", "")).strip()
                if ts_code and ts_code not in seen_codes:
                    seen_codes.add(ts_code)
                    result_rows.append(
                        {
                            "ts_code": ts_code,
                            "trade_date": row.get("trade_date", ""),
                        }
                    )

    if isinstance(result_df, pd.DataFrame) and "ts_code" in result_df.columns:
        for _, row in result_df.iterrows():
            ts_code = str(row.get("ts_code", "")).strip()
            if ts_code and ts_code not in seen_codes:
                seen_codes.add(ts_code)
                result_rows.append(
                    {
                        "ts_code": ts_code,
                        "trade_date": row.get("trade_date", ""),
                    }
                )

    pool_df = pd.DataFrame(result_rows, columns=["ts_code", "trade_date"])
    pool_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已更新股票池: {output_file}")
    return pool_df


def get_daily_basic(pro, ts_code, trade_date):
    """优先读取与主筛选脚本一致的日线基础信息缓存。"""
    return get_shared_daily_basic(pro, ts_code, trade_date)


def check_basic_conditions(pro, ts_code, trade_date):
    """筛选日基本面条件：换手率大于 10%，总市值大于设定阈值。"""
    basic_df = get_daily_basic(pro, ts_code, trade_date)

    if basic_df is None or basic_df.empty:
        return False

    row = basic_df.iloc[0]
    turnover_over_10 = row["turnover_rate"] > 4
    total_mv_over_limit = row["total_mv"] > config.MIN_TOTAL_MV
    #print(f"{ts_code} {trade_date} 换手率: {row['turnover_rate']}, 总市值: {row['total_mv']}")
    return turnover_over_10 and total_mv_over_limit


def check_turnover_over_10(pro, ts_code, trade_date):
    """兼容旧调用：只判断换手率是否大于 10%。"""
    basic_df = get_daily_basic(pro, ts_code, trade_date)

    if basic_df is None or basic_df.empty:
        return False

    return basic_df.iloc[0]["turnover_rate"] > 5


def run_search(
    as_of_date=None,
    result_dir=None,
    stock_file=None,
    pro=None,
    request_sleep=0.08,
    save_file=True,
):
    """执行股票筛选。"""
    pro = pro or create_pro()
    as_of_date = as_of_date or config.get_end_date()
    print(as_of_date)
    result_dir = Path(result_dir or config.RESULT_DIR)
    stock_file = Path(stock_file or config.STOCK_FILE)
    start_date = str(int(as_of_date) - config.SEARCH_DAILY_LOOKBACK_YEARS * 10000)

    number_data = pd.read_csv(stock_file)

    if "ts_code" not in number_data.columns:
        raise ValueError("number.csv 中没有 ts_code 列")

    number_data = filter_non_st_stocks(number_data)
    codes = number_data["ts_code"].dropna().astype(str).str.strip().tolist()
    #print(codes)
    total_codes = len(codes)
    #codes=['002579.SZ']
    result = []
    total_codes = len(codes)
    show_progress(0, total_codes, "search 下载日线")

    for index, ts_code in enumerate(codes, start=1):
        #print(1)
        try:
            #ts_code='600176.SH'
            df, downloaded = get_daily_data(pro, ts_code, start_date, as_of_date)
            if df is None or df.empty:
                continue

            df = df.sort_values("trade_date").reset_index(drop=True)
            last_trade_date = df.iloc[-1]["trade_date"]
            #print(check_basic_conditions(pro, ts_code, last_trade_date))
            #print(match_strategy(df))
            if match_strategy(df) and check_basic_conditions(pro, ts_code, last_trade_date):
                result.append(
                    {
                        "ts_code": ts_code,
                        "trade_date": last_trade_date,
                    }
                )
                print(f"符合条件: {ts_code}")
            
            if downloaded and request_sleep:
                time.sleep(request_sleep)
        except Exception as e:
            print(f"{ts_code} 获取失败: {e}")
            
        finally:
            show_progress(index, total_codes, "search 下载日线")

    result_df = pd.DataFrame(result, columns=["ts_code", "trade_date"])
    result_df = attach_backtest_result(result_df)
    print(result_df)

    if save_file:
        result_dir.mkdir(exist_ok=True)
        output_file = result_dir / f"量化底部筛选结果_{as_of_date}.csv"
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"已保存筛选结果: {output_file}")

        build_pool_from_search_results(result_df)

    return result_df


if __name__ == "__main__":
    run_search()
