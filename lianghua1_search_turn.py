from pathlib import Path

import pandas as pd
import tushare as ts

import config
from progress import show_progress
from stock_filters import filter_non_st_stocks


TOKEN = config.TUSHARE_TOKEN
BASE_DIR = Path(__file__).resolve().parent
SEARCH_CACHE_DIR = BASE_DIR / "model_artifacts" / "download_daily"
DAILY_CACHE_DIR = SEARCH_CACHE_DIR / "daily"
DAILY_BASIC_CACHE_DIR = SEARCH_CACHE_DIR / "daily_basic"
DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
DAILY_BASIC_FIELDS = "ts_code,trade_date,turnover_rate,total_mv"
DOWNLOAD_YEARS = 5
BACKTEST_FORWARD_DAYS = config.BACKTEST_FORWARD_DAYS
MIN_TURN_TOTAL_MV = 900000
MIN_TURNOVER_RATE = 4


def create_pro():
    ts.set_token(config.TUSHARE_TOKEN)
    return ts.pro_api()


def ensure_cache_dirs():
    DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_BASIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_code(ts_code):
    return str(ts_code).replace(".", "_")


def read_cached_csv(cache_file):
    cache_file = Path(cache_file)
    if not cache_file.exists():
        return None
    try:
        return pd.read_csv(cache_file, dtype={"ts_code": str, "trade_date": str})
    except Exception as error:
        print(f"{cache_file} 读取失败，将重新下载: {error}")
        return None


def five_year_start_date(end_date):
    return (pd.to_datetime(str(end_date)) - pd.DateOffset(years=DOWNLOAD_YEARS)).strftime("%Y%m%d")


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


def get_daily_data(pro, ts_code, start_date, end_date):
    ensure_cache_dirs()
    cache_file = DAILY_CACHE_DIR / f"{cache_code(ts_code)}.csv"
    cached = read_cached_csv(cache_file)
    if cached is None or cached.empty:
        raise FileNotFoundError(f"{ts_code} 没有本地日线缓存，请先运行 python download.py")

    cached["trade_date"] = cached["trade_date"].astype(str)
    cached = cached.drop_duplicates(subset=["trade_date"], keep="last")
    cached = cached.sort_values("trade_date").reset_index(drop=True)
    result = cached[(cached["trade_date"] >= str(start_date)) & (cached["trade_date"] <= str(end_date))].copy()
    if result.empty:
        raise ValueError(f"{ts_code} 本地日线缓存不覆盖 {start_date}-{end_date}，请先运行 python download.py")
    return result, False


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
    pct_chg = pd.to_numeric(row.get("pct_chg"), errors="coerce")
    if pd.notna(pct_chg):
        return pct_chg >= config.LIMIT_UP_PCT

    close = pd.to_numeric(row.get("close"), errors="coerce")
    pre_close = pd.to_numeric(row.get("pre_close"), errors="coerce")
    if pd.isna(close) or pd.isna(pre_close) or pre_close <= 0:
        return False
    return close >= pre_close * (1 + config.LIMIT_UP_PCT / 100)


def has_two_day_limit_up_streak_in_recent_40_days(df):
    recent_df = normalize_daily_frame(df).tail(30).reset_index(drop=True)
    if recent_df.empty:
        return False

    current_streak = 0
    i=0
    for _, row in recent_df.iterrows():
        if is_limit_up(row):
            i+=1
            current_streak += 1
            #print(i)
            #print(current_streak,row["trade_date"],row["close"],row["pre_close"])
            if i >= 2:
                
                return True
        else:
            current_streak = 0
    #print(current_streak)
    return False


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


def add_macd(df, fast=12, slow=26, signal=9):
    close = pd.to_numeric(df["close"], errors="coerce")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    df["diff"] = ema_fast - ema_slow
    df["dea"] = df["diff"].ewm(span=signal, adjust=False).mean()
    df["macd"] = (df["diff"] - df["dea"]) * 2
    return df


def volume_above_prev_20d_avg(df, ratio=1.05):
    vol = pd.to_numeric(df["vol"], errors="coerce")
    if len(vol) < 21 or vol.tail(21).isna().any():
        return False

    prev_20d_avg_vol = vol.iloc[-21:-1].mean()
    return vol.iloc[-1] > prev_20d_avg_vol * ratio


def match_strategy(df):
    if df is None or len(df) < max(35, config.MIN_DAILY_ROWS_FOR_SEARCH):
        return False

    df = normalize_daily_frame(df).sort_values("trade_date").reset_index(drop=True)
    if len(df) < 3:
        return False

    df = add_macd(df)
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    day_before_yesterday = df.iloc[-3]

    required_values = [
        today["open"],
        today["high"],
        today["close"],
        today["pct_chg"],
        today["vol"],
        today["diff"],
        today["dea"],
        yesterday["close"],
        yesterday["vol"],
        yesterday["macd"],
        day_before_yesterday["close"],
        day_before_yesterday["vol"],
    ]
    if any(pd.isna(value) for value in required_values):
        return False

    ma30 = df["close"].tail(30).mean()
    ma60 = df["close"].tail(60).mean()
    ma120 = df["close"].tail(120).mean()
    ma5 = df["close"].tail(5).mean()
    ma10 = df["close"].tail(10).mean()
    ma20 = df["close"].tail(20).mean()
    if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma30) or pd.isna(ma60) or pd.isna(ma120):
        return False

    recent_4_close = df["close"].tail(5).reset_index(drop=True)
    if len(recent_4_close) < 5 or recent_4_close.isna().any():
        return False
    recent_4_yin_count = (recent_4_close.diff().tail(4) < 0).sum()
    recent_4_at_least_2_yin = recent_4_yin_count >= 2

    yesterday_ma5 = df["close"].iloc[-6:-1].mean()
    day_before_yesterday_ma5 = df["close"].iloc[-7:-2].mean()
    if pd.isna(yesterday_ma5) or pd.isna(day_before_yesterday_ma5):
        return False
    yesterday_ma5_below_prev_ma5 = yesterday_ma5 < day_before_yesterday_ma5

    has_recent_two_day_limit_up_streak = has_two_day_limit_up_streak_in_recent_40_days(df)
    has_prev_year_limit_up_history = has_limit_up_history_in_prev_months(df)
    yesterday_down = yesterday["close"] < day_before_yesterday["close"]
    yesterday_volume_up = yesterday["vol"] > day_before_yesterday["vol"]
    yesterday_macd_below_zero = yesterday["macd"] < 0
    today_up = today["close"] > yesterday["close"]
    today_volume_up = today["vol"] > yesterday["vol"]
    volume_above_20d_avg = volume_above_prev_20d_avg(df)
    if today["pct_chg"] > 9.85:
        today_volume_up = True
        volume_above_20d_avg = True 
        #print(ma30)
    today_above_ma30_ma60_ma120 = today["close"] > ma30 and today["close"] > ma60 and today["close"] > ma120 and today["close"] > ma20
    high_above_ma5_ma10 = today["high"] > ma5 and today["high"] > ma10
    today_bullish_over_2pct = today["close"] > today["open"] and today["pct_chg"] > 2
    today_diff_dea_above_zero = today["diff"] > 0 and today["dea"] > 0
    
    #print(volume_above_20d_avg,yesterday_ma5_below_prev_ma5,recent_4_at_least_2_yin,has_recent_two_day_limit_up_streak, yesterday_down, yesterday_volume_up, yesterday_macd_below_zero, today_up, today_volume_up, today_above_ma30_ma60_ma120, today_diff_dea_above_zero)
    return (
        yesterday_down
        and yesterday_volume_up
        and yesterday_macd_below_zero
        and today_up
        and today_bullish_over_2pct
        and today_volume_up
        and today_above_ma30_ma60_ma120
        and today_diff_dea_above_zero
        and has_recent_two_day_limit_up_streak
        and has_prev_year_limit_up_history
        and recent_4_at_least_2_yin
        and yesterday_ma5_below_prev_ma5
    )


def build_pool_from_search_results(result_df):
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
    ensure_cache_dirs()
    cache_file = DAILY_BASIC_CACHE_DIR / f"{trade_date}.csv"
    cached = read_cached_csv(cache_file)
    if cached is None or cached.empty:
        raise FileNotFoundError(f"{trade_date} 没有本地 daily_basic 缓存，请先运行 python download.py")
    return cached[cached["ts_code"].astype(str) == str(ts_code)].copy()


def check_basic_conditions(pro, ts_code, trade_date, df):
    basic_df = get_daily_basic(pro, ts_code, trade_date)
    if basic_df is None or basic_df.empty:
        return False

    row = basic_df.iloc[0]
    turnover_rate = pd.to_numeric(row.get("turnover_rate"), errors="coerce")
    total_mv = pd.to_numeric(row.get("total_mv"), errors="coerce")
    if pd.isna(turnover_rate) or pd.isna(total_mv):
        return False

    return turnover_rate > MIN_TURNOVER_RATE and total_mv > MIN_TURN_TOTAL_MV


def run_search(
    as_of_date=None,
    result_dir=None,
    stock_file=None,
    pro=None,
    request_sleep=0.08,
    save_file=True,
):
    pro = pro or create_pro()
    as_of_date = as_of_date or config.get_end_date()
    result_dir = Path(result_dir or config.RESULT_DIR)
    stock_file = Path(stock_file or config.STOCK_FILE)
    start_date = str(int(as_of_date) - config.SEARCH_DAILY_LOOKBACK_YEARS * 10000)
    print(as_of_date)

    number_data = pd.read_csv(stock_file)
    if "ts_code" not in number_data.columns:
        raise ValueError("number.csv 中没有 ts_code 列")

    number_data = filter_non_st_stocks(number_data)
    name_map = {}
    if "name" in number_data.columns:
        name_map = dict(zip(number_data["ts_code"].astype(str), number_data["name"].astype(str)))

    codes = number_data["ts_code"].dropna().astype(str).str.strip().tolist()
    result = []
    total_codes = len(codes)
    show_progress(0, total_codes, "turn search 读取日线")

    for index, ts_code in enumerate(codes, start=1):
        try:
            #ts_code='000811.SZ'
            df, _ = get_daily_data(pro, ts_code, start_date, as_of_date)
            if df is None or df.empty:
                continue

            df = df.sort_values("trade_date").reset_index(drop=True)
            last_trade_date = df.iloc[-1]["trade_date"]


            #print(match_strategy(df))
            #print(check_basic_conditions(pro, ts_code, last_trade_date, df))
            if match_strategy(df) and check_basic_conditions(pro, ts_code, last_trade_date, df):
                last_day = df.iloc[-1]
                result.append(
                    {
                        "ts_code": ts_code,
                        "name": name_map.get(ts_code, ""),
                        "trade_date": last_trade_date,
                        "open": last_day.get("open", ""),
                        "close": last_day.get("close", ""),
                        "pct_chg": last_day.get("pct_chg", ""),
                        "vol": last_day.get("vol", ""),
                    }
                )
                print(f"符合条件: {ts_code}")

        except Exception as e:
            print(f"{ts_code} 获取失败: {e}")
        finally:
            show_progress(index, total_codes, "turn search 读取日线")

    result_df = pd.DataFrame(
        result,
        columns=["ts_code", "name", "trade_date", "open", "close", "pct_chg", "vol"],
    )
    result_df = attach_backtest_result(result_df)
    print(result_df)

    if save_file:
        result_dir.mkdir(exist_ok=True)
        output_file = result_dir / f"反转筛选结果_{as_of_date}.csv"
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"已保存反转筛选结果: {output_file}")
        build_pool_from_search_results(result_df)

    return result_df


def run_search_turn(*args, **kwargs):
    return run_search(*args, **kwargs)


def run_search_continuered(*args, **kwargs):
    return run_search(*args, **kwargs)


if __name__ == "__main__":
    run_search()
