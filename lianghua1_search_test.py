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


def has_limit_up_streak_between_3_and_5(df):
    recent_df = normalize_daily_frame(df).tail(10).reset_index(drop=True)
    if len(recent_df) < config.MIN_LIMIT_UP_STREAK:
        return False

    current_streak = 0
    for _, row in recent_df.iterrows():
        if is_limit_up(row):
            current_streak += 1
            continue

        if 2 <= current_streak <= config.MAX_LIMIT_UP_STREAK:
            return True
        current_streak = 0

    return 2 <= current_streak <= config.MAX_LIMIT_UP_STREAK


def has_bad_limit_break_after_streak(df):
    data = normalize_daily_frame(df).sort_values("trade_date").reset_index(drop=True)
    if len(data) < config.MIN_LIMIT_UP_STREAK + 1:
        return False

    current_streak = 0
    for index, row in data.iterrows():
        if is_limit_up(row):
            current_streak += 1
            continue

        if current_streak >= config.MIN_LIMIT_UP_STREAK:
            open_price = pd.to_numeric(row.get("open"), errors="coerce")
            close_price = pd.to_numeric(row.get("close"), errors="coerce")
            pre_close = pd.to_numeric(row.get("pre_close"), errors="coerce")
            if pd.isna(pre_close) and index > 0:
                pre_close = pd.to_numeric(data.iloc[index - 1].get("close"), errors="coerce")

            if (
                pd.notna(open_price)
                and pd.notna(close_price)
                and pd.notna(pre_close)
                and open_price > 0
                and pre_close > 0
                and open_price / pre_close - 1 > 0.05
                and close_price / open_price - 1 < -0.105
            ):
                return True

        current_streak = 0

    return False


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


def is_bullish_over_3pct(row):
    open_price = pd.to_numeric(row.get("open"), errors="coerce")
    close_price = pd.to_numeric(row.get("close"), errors="coerce")
    pct_chg = pd.to_numeric(row.get("pct_chg"), errors="coerce")
    if pd.isna(open_price) or pd.isna(close_price) or pd.isna(pct_chg):
        return False
    return close_price > open_price and pct_chg > 3


def match_strategy(df):
    if df is None or len(df) < max(121, config.MIN_DAILY_ROWS_FOR_SEARCH):
        return False

    df = normalize_daily_frame(df).sort_values("trade_date").reset_index(drop=True)
    if len(df) < 3:
        return False
    if has_bad_limit_break_after_streak(df):
        return False

    last_day = df.iloc[-1]
    prev_day = df.iloc[-2]
    prev_prev_day = df.iloc[-3]
    last_bullish_over_3pct = is_bullish_over_3pct(last_day)

    has_limit_up_streak = has_limit_up_streak_between_3_and_5(df)
    cc=prev_prev_day['close']>prev_day['close']
    
    #print(last_day['close'])
    #last_not_limit_up = not is_limit_up(last_day)
    last_is_yang = last_day['close'] / prev_day['close'] - 1 > -0.0025
    last_volume_up = last_day['vol'] > prev_day['vol']
    last_volume_not_too_low = (
        last_day['vol'] >= df.tail(config.RECENT_VOLUME_WINDOW)['vol'].max() * config.MIN_LAST_VOLUME_PEAK_RATIO
    )
    prev_is_yin = (prev_prev_day['close'] > prev_day['close']) or (prev_day['open']>prev_day['close'])
    prev_volume_down = prev_day['vol'] < prev_prev_day['vol']
    last_close_above_ma = last_day['close'] > df.tail(config.MA_DAYS_FOR_SEARCH)['close'].mean()
    moving_average_upward = ma60_ma120_upward(df)
    volume_above_20d_avg = volume_above_prev_20d_avg(df)
    if last_day['pct_chg'] >9.9:
        last_volume_up = True
        last_volume_not_too_low = True
        volume_above_20d_avg = True

    #print(has_limit_up_streak,last_is_yang, last_volume_up,prev_is_yin,prev_volume_down,last_close_above_ma,cc)
    return (
        has_limit_up_streak
        and last_is_yang
        and last_volume_up
        and last_volume_not_too_low
        and prev_is_yin
        and prev_volume_down
        and last_close_above_ma
        and moving_average_upward
        and volume_above_20d_avg
        and last_bullish_over_3pct
        and cc
    )


def match_volume_price_strategy(df):
    if df is None or len(df) < max(121, config.MIN_DAILY_ROWS_FOR_SEARCH):
        return False

    df = normalize_daily_frame(df).sort_values("trade_date").reset_index(drop=True)
    if len(df) < 3:
        return False
    if has_bad_limit_break_after_streak(df):
        return False

    if not ma60_ma120_upward(df):
        return False

    if not volume_above_prev_20d_avg(df):
        return False

    if not has_limit_up_streak_between_3_and_5(df):
        return False

    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    day_before_yesterday = df.iloc[-3]

    if not is_limit_up(day_before_yesterday):
        return False
    if is_limit_up(yesterday):
        return False
    
    today_pct = is_bullish_over_3pct(today) and today["pct_chg"] < 9.9
    

    yesterday_vol = pd.to_numeric(yesterday.get("vol"), errors="coerce")
    prev_vol = pd.to_numeric(day_before_yesterday.get("vol"), errors="coerce")
    yesterday_close = pd.to_numeric(yesterday.get("close"), errors="coerce")
    prev_close = pd.to_numeric(day_before_yesterday.get("close"), errors="coerce")
    today_open = pd.to_numeric(today.get("open"), errors="coerce")
    today_close = pd.to_numeric(today.get("close"), errors="coerce")

    if any(pd.isna(x) for x in (yesterday_vol, prev_vol, yesterday_close, prev_close, today_open, today_close)):
        return False

    return (
        yesterday_vol > prev_vol
        and yesterday_close > prev_close
        and today_open > yesterday_close
        and today_close > yesterday_close
        and today_pct
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


def check_basic_conditions(pro, ts_code, trade_date,df):
    basic_df = get_daily_basic(pro, ts_code, trade_date)
    if basic_df is None or basic_df.empty:
        return False

    row = basic_df.iloc[0]
    turnover_over_10 = row["turnover_rate"] > config.MIN_TURNOVER_RATE
    last_day = df.iloc[-1]
    rows=basic_df.head(5)
    #print(rows)
    if last_day['pct_chg']>9.85 and rows["turnover_rate"].max()>config.MIN_TURNOVER_RATE: 
        turnover_over_10=True
    total_mv_over_limit = row["total_mv"] > config.MIN_TOTAL_MV
    return turnover_over_10 and total_mv_over_limit


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
    codes = number_data["ts_code"].dropna().astype(str).str.strip().tolist()
    result = []
    total_codes = len(codes)
    show_progress(0, total_codes, "search 下载日线")

    for index, ts_code in enumerate(codes, start=1):
        try:
            #ts_code='000887.SZ'
            #print(1)
            df, _ = get_daily_data(pro, ts_code, start_date, as_of_date)
            if df is None or df.empty:
                continue

            df = df.sort_values("trade_date").reset_index(drop=True)
            last_trade_date = df.iloc[-1]["trade_date"]
            #print(match_strategy(df))
            #print(check_basic_conditions(pro, ts_code, last_trade_date,df))
            if match_strategy(df) and check_basic_conditions(pro, ts_code, last_trade_date,df):
                result.append(
                    {
                        "ts_code": ts_code,
                        "trade_date": last_trade_date,
                    }
                )
                print(f"符合条件: {ts_code}")
            
            if match_volume_price_strategy(df) and check_basic_conditions(pro, ts_code, last_trade_date,df):
                result.append(
                    {
                        "ts_code": ts_code,
                        "trade_date": last_trade_date,
                    }
                )
                print(f"符合条件: {ts_code}")

        except Exception as e:
            print(f"{ts_code} 获取失败: {e}")
        finally:
            show_progress(index, total_codes, "search 下载日线")

    result_df = pd.DataFrame(result, columns=["ts_code", "trade_date"])
    result_df = attach_backtest_result(result_df)
    print(result_df)

    if save_file:
        result_dir.mkdir(exist_ok=True)
        output_file = result_dir / f"量化筛选结果{as_of_date}.csv"
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"已保存筛选结果: {output_file}")
        build_pool_from_search_results(result_df)

    return result_df


if __name__ == "__main__":
    run_search()
