import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import tushare as ts

import config
from progress import show_progress


BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "model_artifacts" / "download_daily"
DAILY_CACHE_DIR = DOWNLOAD_DIR / "daily"
DAILY_BASIC_CACHE_DIR = DOWNLOAD_DIR / "daily_basic"

START_DATE = "20200101"
REQUEST_SLEEP = 0.08
DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
DAILY_BASIC_FIELDS = "ts_code,trade_date,turnover_rate,total_mv"


def create_pro():
    ts.set_token(config.TUSHARE_TOKEN)
    return ts.pro_api()


def ensure_dirs():
    DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_BASIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_code(ts_code):
    return str(ts_code).replace(".", "_")


def load_stock_codes(stock_file=None):
    stock_file = Path(stock_file or config.STOCK_FILE)
    stock_df = pd.read_csv(stock_file, dtype={"ts_code": str})
    if "ts_code" not in stock_df.columns:
        raise ValueError("number.csv 缺少 ts_code 列")
    return stock_df["ts_code"].dropna().astype(str).str.strip().drop_duplicates().tolist()


def read_cached_csv(cache_file):
    cache_file = Path(cache_file)
    if not cache_file.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(cache_file, dtype={"ts_code": str, "trade_date": str})
    except Exception as error:
        print(f"{cache_file} 读取失败，将重新下载: {error}")
        return pd.DataFrame()


def cache_covers(df, start_date, end_date):
    """判断缓存是否已覆盖请求区间；边界和重复日期统一按 YYYYMMDD 比较。"""
    if df is None or df.empty or "trade_date" not in df.columns:
        return False
    start = pd.to_datetime(str(start_date), format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(str(end_date), format="%Y%m%d", errors="coerce")
    dates = pd.to_datetime(
        df["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d", errors="coerce",
    ).dropna()
    if pd.isna(start) or pd.isna(end) or dates.empty:
        return False
    return dates.min() <= start and dates.max() >= end


def get_end_date():
    return datetime.today().strftime("%Y%m%d")


def download_daily_for_stock(pro, ts_code, start_date, end_date):
    cache_file = DAILY_CACHE_DIR / f"{cache_code(ts_code)}.csv"
    cached = read_cached_csv(cache_file)
    if cache_covers(cached, start_date, end_date):
        print(f"{ts_code} 已有日线缓存 {start_date}-{end_date}，跳过下载")
        return False

    daily = pro.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields=DAILY_FIELDS,
    )
    if daily is None or daily.empty:
        return True

    daily["trade_date"] = daily["trade_date"].astype(str)
    merged = pd.concat([cached, daily], ignore_index=True)
    merged = merged.drop_duplicates(subset=["trade_date"], keep="last")
    merged = merged.sort_values("trade_date").reset_index(drop=True)
    merged.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return True


def download_daily_basic_for_dates(pro, trade_dates):
    total = len(trade_dates)
    show_progress(0, total, "download daily_basic")
    for index, trade_date in enumerate(trade_dates, start=1):
        cache_file = DAILY_BASIC_CACHE_DIR / f"{trade_date}.csv"
        cached = read_cached_csv(cache_file)
        if cached is None or cached.empty:
            basic = pro.daily_basic(
                trade_date=trade_date,
                fields=DAILY_BASIC_FIELDS,
            )
            if basic is not None and not basic.empty:
                basic.to_csv(cache_file, index=False, encoding="utf-8-sig")
            time.sleep(REQUEST_SLEEP)
        else:
            # 每个交易日的基础数据独立缓存；存在有效内容时不重复请求。
            pass

        show_progress(index, total, "download daily_basic")


def collect_trade_dates_from_daily_cache(start_date, end_date):
    trade_dates = set()
    for cache_file in DAILY_CACHE_DIR.glob("*.csv"):
        cached = read_cached_csv(cache_file)
        if cached is not None and not cached.empty and "trade_date" in cached.columns:
            dates = cached["trade_date"].astype(str)
            trade_dates.update(dates[(dates >= start_date) & (dates <= end_date)].tolist())
    return sorted(trade_dates)


def main():
    ensure_dirs()
    pro = create_pro()
    end_date = get_end_date()
    stock_codes = load_stock_codes()

    print(f"[download] daily range: {START_DATE}-{end_date}, stocks={len(stock_codes)}")
    show_progress(0, len(stock_codes), "download daily")
    for index, ts_code in enumerate(stock_codes, start=1):
        try:
            downloaded = download_daily_for_stock(pro, ts_code, START_DATE, end_date)
            if downloaded:
                time.sleep(REQUEST_SLEEP)
        except Exception as error:
            print(f"{ts_code} 日线下载失败: {error}")
        finally:
            show_progress(index, len(stock_codes), "download daily")

    trade_dates = collect_trade_dates_from_daily_cache(START_DATE, end_date)
    print(f"[download] daily_basic dates={len(trade_dates)}")
    download_daily_basic_for_dates(pro, trade_dates)
    print(f"[download] saved to: {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()
