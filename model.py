import json
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import tushare as ts
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

import config
from progress import show_progress


# =====================
# 日期与样本切分参数
# =====================

START_DATE = "20160101"  # 训练行情数据起始日期。
END_DATE = "20251231"  # 训练行情数据结束日期。
UNIVERSE_BENCHMARK_DATE = "20251231"  # 股票池市值过滤使用的基准日期。
TRAIN_END_DATE = "20251231"  # 训练集结束日期。
VALID_END_DATE = "20260331"  # 验证集结束日期。
TEST_END_DATE = "20260624"  # 测试集结束日期。
SPLIT_GAP_DAYS = 15  # 训练/验证/测试之间留出的时间隔离天数，降低未来泄漏。


# =====================
# lianghua1_search 策略样本参数
# =====================

SEARCH_MA_DAYS = 120  # 只保留 120 日均线方向过滤。
SEARCH_LOOKBACK_ROWS = SEARCH_MA_DAYS + 1  # 比较当前 120 日均线和上一交易日 120 日均线。
BREAKOUT_LOOKAHEAD_DAYS = 3  # 突破概率标签观察未来多少个交易日。
BREAKOUT_BUFFER = 0.015  # 突破箱体上沿的缓冲幅度，0.015 表示超过上沿 1.5%。
UPSIDE_LOOKAHEAD_DAYS = 5  # 上涨空间标签观察未来多少个交易日。


# =====================
# 一字板过滤参数
# =====================

ONE_WORD_LIMIT_LOOKBACK_DAYS = 20  # 训练样本中过去多少日内出现一字板就剔除。
ONE_WORD_PRICE_EPS = 1e-4  # 判断开高低收几乎相等时允许的价格误差。
ONE_WORD_LIMIT_PCT = 9.8  # 判断涨跌停幅度的阈值，9.8 表示接近 10% 涨跌停。


# =====================
# 特征与模型参数
# =====================

QUANTILES = (0.1, 0.5, 0.9)  # 上涨空间分位数预测：保守/中位/乐观。
ROLLING_WINDOWS = (3, 5, 10, 20, 60, 75)  # 通用滚动特征窗口。
MAX_TRAIN_ROWS = 2_000_000  # 最大训练样本数，超过后随机抽样。
RANDOM_STATE = 42  # 随机种子，保证抽样和模型训练尽量可复现。


# =====================
# Tushare 与下载参数
# =====================

REQUEST_SLEEP = 0.03  # 每次下载请求后的暂停秒数，用来降低接口压力。
DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"  # 日线行情字段。
DAILY_BASIC_FIELDS = "ts_code,trade_date,turnover_rate,total_mv"  # daily_basic 字段：换手率和总市值。


# =====================
# 股票池过滤参数
# =====================

UNIVERSE_MIN_TOTAL_MV = 700_000  # 最小总市值，单位万元；700000 = 70 亿。
UNIVERSE_MAX_TOTAL_MV = 10_000_000  # 最大总市值，单位万元；10000000 = 1000 亿。


# =====================
# GPU 训练参数
# =====================

USE_GPU = True  # 是否优先使用 LightGBM GPU 训练。
GPU_DEVICE_TYPE = "gpu"  # LightGBM 设备类型，通常为 gpu。
GPU_PLATFORM_ID = 0  # GPU 平台编号，多平台环境才需要调整。
GPU_DEVICE_ID = 0  # GPU 设备编号，单卡通常为 0。
GPU_USE_DP = False  # 是否使用双精度，False 通常更快。
FALLBACK_TO_CPU_IF_GPU_FAILS = True  # GPU 不可用时是否自动回退到 CPU。


# =====================
# 文件路径参数
# =====================

BASE_DIR = Path(__file__).resolve().parent  # 项目根目录。
MODEL_DIR = BASE_DIR / "model_artifacts"  # 模型、缓存、预测结果保存目录。
CACHE_DIR = MODEL_DIR / "tushare_by_date"  # 按交易日缓存 Tushare 数据的目录。
PANEL_CACHE_FILE = MODEL_DIR / "panel_2016_2025.csv"  # 合并后的全量面板缓存文件。
MODEL_BUNDLE_FILE = MODEL_DIR / "short_breakout_bundle.pkl"  # 训练后的模型 bundle。
METRICS_FILE = MODEL_DIR / "short_breakout_metrics.json"  # 测试集评估指标输出文件。
LATEST_PRED_FILE = MODEL_DIR / "latest_predictions.csv"  # 最新预测结果输出文件。


LAG_WINDOWS = (1, 2, 3, 5, 10)
FEATURE_CACHE_FILE = MODEL_DIR / "feature_cache.csv"


def create_pro():
    ts.set_token(config.TUSHARE_TOKEN)
    return ts.pro_api()


def ensure_dirs():
    MODEL_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)


def load_stock_universe(stock_file=None):
    stock_file = Path(stock_file or config.STOCK_FILE)
    stock_df = pd.read_csv(stock_file, dtype={"ts_code": str, "name": str})
    if "ts_code" not in stock_df.columns:
        raise ValueError("number.csv 缺少 ts_code 列")

    stock_df["ts_code"] = stock_df["ts_code"].astype(str).str.strip()
    if "name" in stock_df.columns:
        stock_df = stock_df[~stock_df["name"].astype(str).str.contains("ST", case=False, na=False)]

    stock_df = stock_df[stock_df["ts_code"].astype(str).str.len() > 0].copy()

    pro = create_pro()
    benchmark_basic = pro.daily_basic(
        trade_date=UNIVERSE_BENCHMARK_DATE,
        fields="ts_code,trade_date,total_mv",
    )
    if benchmark_basic is None or benchmark_basic.empty:
        raise ValueError(f"{UNIVERSE_BENCHMARK_DATE} 没有取到 daily_basic 数据")

    benchmark_basic["ts_code"] = benchmark_basic["ts_code"].astype(str).str.strip()
    benchmark_basic["total_mv"] = pd.to_numeric(benchmark_basic["total_mv"], errors="coerce")
    benchmark_basic = benchmark_basic[
        (benchmark_basic["total_mv"] >= UNIVERSE_MIN_TOTAL_MV)
        & (benchmark_basic["total_mv"] <= UNIVERSE_MAX_TOTAL_MV)
    ].copy()

    universe = stock_df.merge(benchmark_basic[["ts_code"]], on="ts_code", how="inner")
    universe = universe["ts_code"].dropna().astype(str).str.strip().unique().tolist()
    universe = sorted(universe)
    print(
        f"[universe] benchmark={UNIVERSE_BENCHMARK_DATE}, "
        f"mv_range=[{UNIVERSE_MIN_TOTAL_MV}, {UNIVERSE_MAX_TOTAL_MV}], "
        f"stocks={len(universe)}"
    )
    return universe


def get_trade_dates(pro, start_date=START_DATE, end_date=END_DATE):
    cal = pro.trade_cal(
        exchange="SSE",
        is_open="1",
        start_date=start_date,
        end_date=end_date,
        fields="cal_date",
    )
    if cal is None or cal.empty:
        raise ValueError("trade_cal 未返回可用交易日")
    return sorted(cal["cal_date"].astype(str).tolist())


def _read_cached_day(cache_file):
    if not cache_file.exists():
        return None
    day_df = pd.read_csv(cache_file, dtype={"ts_code": str, "trade_date": str})
    return day_df


def get_cached_trade_dates(cache_dir=CACHE_DIR):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return set()

    cached_dates = set()
    for file in cache_dir.glob("*.csv"):
        if file.stem.isdigit() and len(file.stem) == 8:
            cached_dates.add(file.stem)
    return cached_dates


def fetch_trade_date_panel(pro, trade_date, allowed_codes, request_sleep=REQUEST_SLEEP):
    cache_file = CACHE_DIR / f"{trade_date}.csv"
    cached = _read_cached_day(cache_file)
    if cached is not None:
        return cached

    daily = pro.daily(trade_date=trade_date, fields=DAILY_FIELDS)
    basic = pro.daily_basic(trade_date=trade_date, fields=DAILY_BASIC_FIELDS)

    if daily is None or daily.empty:
        return pd.DataFrame()

    if basic is None or basic.empty:
        basic = pd.DataFrame(columns=["ts_code", "trade_date", "turnover_rate", "total_mv"])

    merged = daily.merge(basic, on=["ts_code", "trade_date"], how="left")
    merged = merged[merged["ts_code"].isin(allowed_codes)].copy()
    if merged.empty:
        return merged

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "turnover_rate",
        "total_mv",
    ]
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").astype("float32")

    merged = merged.sort_values("ts_code").reset_index(drop=True)
    merged.to_csv(cache_file, index=False, encoding="utf-8-sig")

    if request_sleep:
        time.sleep(request_sleep)

    return merged


def load_or_download_panel(pro, stock_codes, start_date=START_DATE, end_date=END_DATE):
    trade_dates = get_trade_dates(pro, start_date=start_date, end_date=end_date)
    allowed_codes = set(stock_codes)
    trade_dates_set = set(trade_dates)

    if PANEL_CACHE_FILE.exists():
        panel = pd.read_csv(PANEL_CACHE_FILE, dtype={"ts_code": str, "trade_date": str})
        panel_dates = set(panel["trade_date"].astype(str).unique().tolist())
        if trade_dates_set.issubset(panel_dates):
            print("[data] using panel cache")
            return panel[panel["ts_code"].isin(allowed_codes)].copy()

    cached_dates = get_cached_trade_dates()
    missing_dates = [trade_date for trade_date in trade_dates if trade_date not in cached_dates]

    print(
        f"[data] cached dates: {len(trade_dates) - len(missing_dates)}/{len(trade_dates)}, "
        f"missing: {len(missing_dates)}"
    )

    total_missing = len(missing_dates)
    show_progress(0, total_missing, "model 下载缺失日线")
    for index, trade_date in enumerate(missing_dates, start=1):
        fetch_trade_date_panel(pro, trade_date, allowed_codes)
        show_progress(index, total_missing, "model 下载缺失日线")

        if index == 1 or index % 20 == 0 or index == total_missing:
            print(f"[data] downloaded missing dates: {index}/{total_missing}")

    day_frames = []
    total_days = len(trade_dates)
    show_progress(0, total_days, "model 加载缓存日线")
    for index, trade_date in enumerate(trade_dates, start=1):
        cache_file = CACHE_DIR / f"{trade_date}.csv"
        day_df = _read_cached_day(cache_file)
        if day_df is not None and not day_df.empty:
            day_df = day_df[day_df["ts_code"].isin(allowed_codes)].copy()
            if not day_df.empty:
                day_frames.append(day_df)

        show_progress(index, total_days, "model 加载缓存日线")
        if index == 1 or index % 100 == 0 or index == total_days:
            print(f"[data] loading cached dates: {index}/{total_days}")

    if not day_frames:
        raise ValueError("未下载到任何训练数据")

    panel = pd.concat(day_frames, ignore_index=True)
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    panel.to_csv(PANEL_CACHE_FILE, index=False, encoding="utf-8-sig")
    return panel


def _safe_ratio(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def is_limit_up_row(row):
    pct_chg = pd.to_numeric(row.get("pct_chg"), errors="coerce")
    if pd.notna(pct_chg) and pct_chg >= config.LIMIT_UP_PCT:
        return True

    close = pd.to_numeric(row.get("close"), errors="coerce")
    pre_close = pd.to_numeric(row.get("pre_close"), errors="coerce")
    if pd.isna(close) or pd.isna(pre_close) or pre_close <= 0:
        return False

    return close >= pre_close * (1 + config.LIMIT_UP_PCT / 100)


def row_matches_search_strategy(window_df):
    if window_df is None or len(window_df) < SEARCH_LOOKBACK_ROWS:
        return False

    window_df = window_df.sort_values("trade_date").reset_index(drop=True)
    close = pd.to_numeric(window_df["close"], errors="coerce")
    current_ma = close.tail(SEARCH_MA_DAYS).mean()
    previous_ma = close.iloc[-(SEARCH_MA_DAYS + 1) : -1].mean()

    if pd.isna(current_ma) or pd.isna(previous_ma):
        return False

    return current_ma >= previous_ma


def mark_search_strategy_windows(data):
    data["is_search_setup"] = 0
    grouped = data.groupby("ts_code", group_keys=False)
    groups = list(grouped)
    total_groups = len(groups)
    show_progress(0, total_groups, "model 滚动策略样本")

    for group_index, (ts_code, group_df) in enumerate(groups, start=1):
        group_df = group_df.sort_values("trade_date").reset_index()
        setup_flags = np.zeros(len(group_df), dtype="int8")
        for idx in range(SEARCH_LOOKBACK_ROWS - 1, len(group_df)):
            window_df = group_df.iloc[: idx + 1].copy()
            if row_matches_search_strategy(window_df):
                setup_flags[idx] = 1
        data.loc[group_df["index"], "is_search_setup"] = setup_flags
        show_progress(group_index, total_groups, "model 滚动策略样本")

    data["is_search_setup"] = data["is_search_setup"].astype("int8")
    return data


def get_feature_columns():
    feature_columns = [
        "open_rel",
        "close_rel",
        "high_rel",
        "low_rel",
        "log_vol",
        "turnover_rate",
        "log_mv",
        "body_pct",
        "hl_range",
        "upper_shadow",
        "lower_shadow",
        "daily_ret",
    ]

    for window in ROLLING_WINDOWS:
        feature_columns.extend(
            [
                f"ret_mean_{window}",
                f"ret_std_{window}",
                f"close_ma_gap_{window}",
                f"vol_ma_gap_{window}",
                f"vol_std_{window}",
                f"turnover_mean_{window}",
                f"turnover_std_{window}",
                f"mv_ma_gap_{window}",
            ]
        )

    for lag in LAG_WINDOWS:
        feature_columns.extend(
            [
                f"ret_lag_{lag}",
                f"vol_chg_{lag}",
                f"turnover_chg_{lag}",
            ]
        )

    return list(dict.fromkeys(feature_columns))


def prepare_panel_for_strategy(panel):
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "turnover_rate",
        "total_mv",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")
    return data


def build_feature_frames_uncached(panel, mark_all_search_windows=True):
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "turnover_rate",
        "total_mv",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")

    grouped = data.groupby("ts_code", group_keys=False)
    if mark_all_search_windows:
        data = mark_search_strategy_windows(data)
    else:
        data["is_search_setup"] = 0

    data["open_rel"] = _safe_ratio(data["open"], data["pre_close"]) - 1
    data["close_rel"] = _safe_ratio(data["close"], data["pre_close"]) - 1
    data["high_rel"] = _safe_ratio(data["high"], data["pre_close"]) - 1
    data["low_rel"] = _safe_ratio(data["low"], data["pre_close"]) - 1
    data["log_vol"] = np.log1p(data["vol"])
    data["log_mv"] = np.log1p(data["total_mv"])
    data["body_pct"] = _safe_ratio(data["close"] - data["open"], data["open"])
    data["hl_range"] = _safe_ratio(data["high"] - data["low"], data["low"])
    data["upper_shadow"] = _safe_ratio(
        data["high"] - data[["open", "close"]].max(axis=1),
        data["pre_close"],
    )
    data["lower_shadow"] = _safe_ratio(
        data[["open", "close"]].min(axis=1) - data["low"],
        data["pre_close"],
    )
    data["daily_ret"] = data["pct_chg"] / 100.0

    feature_columns = [
        "open_rel",
        "close_rel",
        "high_rel",
        "low_rel",
        "log_vol",
        "turnover_rate",
        "log_mv",
        "body_pct",
        "hl_range",
        "upper_shadow",
        "lower_shadow",
        "daily_ret",
    ]

    show_progress(0, len(ROLLING_WINDOWS), "model 构造滚动特征")
    for window_index, window in enumerate(ROLLING_WINDOWS, start=1):
        ret_roll = grouped["daily_ret"].transform(lambda s: s.rolling(window).mean())
        ret_std = grouped["daily_ret"].transform(lambda s: s.rolling(window).std())
        close_ma = grouped["close"].transform(lambda s: s.rolling(window).mean())
        vol_ma = grouped["vol"].transform(lambda s: s.rolling(window).mean())
        vol_std = grouped["vol"].transform(lambda s: s.rolling(window).std())
        turnover_mean = grouped["turnover_rate"].transform(lambda s: s.rolling(window).mean())
        turnover_std = grouped["turnover_rate"].transform(lambda s: s.rolling(window).std())
        mv_ma = grouped["total_mv"].transform(lambda s: s.rolling(window).mean())

        data[f"ret_mean_{window}"] = ret_roll
        data[f"ret_std_{window}"] = ret_std
        data[f"close_ma_gap_{window}"] = _safe_ratio(data["close"], close_ma) - 1
        data[f"vol_ma_gap_{window}"] = _safe_ratio(data["vol"], vol_ma) - 1
        data[f"vol_std_{window}"] = vol_std
        data[f"turnover_mean_{window}"] = turnover_mean
        data[f"turnover_std_{window}"] = turnover_std
        data[f"mv_ma_gap_{window}"] = _safe_ratio(data["total_mv"], mv_ma) - 1

        feature_columns.extend(
            [
                f"ret_mean_{window}",
                f"ret_std_{window}",
                f"close_ma_gap_{window}",
                f"vol_ma_gap_{window}",
                f"vol_std_{window}",
                f"turnover_mean_{window}",
                f"turnover_std_{window}",
                f"mv_ma_gap_{window}",
            ]
        )
        show_progress(window_index, len(ROLLING_WINDOWS), "model 构造滚动特征")

    lag_windows = LAG_WINDOWS
    show_progress(0, len(lag_windows), "model 构造滞后特征")
    for lag_index, lag in enumerate(lag_windows, start=1):
        data[f"ret_lag_{lag}"] = grouped["daily_ret"].shift(lag)
        data[f"vol_chg_{lag}"] = _safe_ratio(data["vol"], grouped["vol"].shift(lag)) - 1
        data[f"turnover_chg_{lag}"] = _safe_ratio(data["turnover_rate"], grouped["turnover_rate"].shift(lag)) - 1
        feature_columns.extend(
            [
                f"ret_lag_{lag}",
                f"vol_chg_{lag}",
                f"turnover_chg_{lag}",
            ]
        )
        show_progress(lag_index, len(lag_windows), "model 构造滞后特征")

    future_breakout_high = grouped["high"].transform(
        lambda s: s.iloc[::-1].shift(1).rolling(BREAKOUT_LOOKAHEAD_DAYS, min_periods=1).max().iloc[::-1]
    )
    future_upside_high = grouped["high"].transform(
        lambda s: s.iloc[::-1].shift(1).rolling(UPSIDE_LOOKAHEAD_DAYS, min_periods=1).max().iloc[::-1]
    )
    data["breakout_label"] = (
        future_breakout_high >= data["close"] * (1 + BREAKOUT_BUFFER)
    ).astype("float32")
    data["upside_5d"] = future_upside_high / data["close"] - 1

    filter_columns = ["is_search_setup"]
    keep_columns = (
        ["ts_code", "trade_date"]
        + filter_columns
        + ["breakout_label", "upside_5d"]
        + feature_columns
    )
    keep_columns = list(dict.fromkeys(keep_columns))
    data = data[keep_columns].replace([np.inf, -np.inf], np.nan)
    for column in feature_columns + ["breakout_label", "upside_5d"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")

    prediction_frame = data.dropna(subset=feature_columns).reset_index(drop=True)
    setup_rows = int((prediction_frame["is_search_setup"] == 1).sum())
    print(
        f"[feature] prediction rows: {len(prediction_frame)}, "
        f"search setup rows: {setup_rows}"
    )

    training_frame = prediction_frame[prediction_frame["is_search_setup"] == 1].dropna(
        subset=["breakout_label", "upside_5d"]
    ).reset_index(drop=True)
    print(f"[feature] labeled search setup rows: {len(training_frame)}")

    feature_columns = list(dict.fromkeys(feature_columns))
    return training_frame, prediction_frame, feature_columns


def is_feature_cache_valid(cached, panel):
    required_columns = set(["ts_code", "trade_date", "breakout_label", "upside_5d"] + get_feature_columns())
    if cached is None or cached.empty or not required_columns.issubset(cached.columns):
        return False

    panel_keys = panel[["ts_code", "trade_date"]].copy()
    panel_keys["ts_code"] = panel_keys["ts_code"].astype(str)
    panel_keys["trade_date"] = pd.to_datetime(panel_keys["trade_date"])
    panel_keys = panel_keys.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    cached_keys = cached[["ts_code", "trade_date"]].copy()
    cached_keys["ts_code"] = cached_keys["ts_code"].astype(str)
    cached_keys["trade_date"] = pd.to_datetime(cached_keys["trade_date"])
    cached_keys = cached_keys.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    return panel_keys.equals(cached_keys)


def load_or_build_feature_cache(panel):
    feature_columns = get_feature_columns()

    if FEATURE_CACHE_FILE.exists():
        cached = pd.read_csv(FEATURE_CACHE_FILE, dtype={"ts_code": str, "trade_date": str})
        if is_feature_cache_valid(cached, panel):
            cached["trade_date"] = pd.to_datetime(cached["trade_date"])
            for column in feature_columns + ["breakout_label", "upside_5d"]:
                cached[column] = pd.to_numeric(cached[column], errors="coerce").astype("float32")
            print(f"[feature] using feature cache: {FEATURE_CACHE_FILE}")
            return cached, feature_columns

        print("[feature] feature cache is stale, rebuilding")

    _, prediction_frame, feature_columns = build_feature_frames_uncached(
        panel,
        mark_all_search_windows=False,
    )
    cache_columns = ["ts_code", "trade_date", "breakout_label", "upside_5d"] + feature_columns
    feature_frame = prediction_frame[cache_columns].copy()
    FEATURE_CACHE_FILE.parent.mkdir(exist_ok=True)
    feature_frame.to_csv(FEATURE_CACHE_FILE, index=False, encoding="utf-8-sig")
    print(f"[feature] saved feature cache: {FEATURE_CACHE_FILE}")
    return feature_frame, feature_columns


def build_feature_frames(panel, mark_all_search_windows=True):
    feature_frame, feature_columns = load_or_build_feature_cache(panel)
    setup_frame = prepare_panel_for_strategy(panel)

    if mark_all_search_windows:
        setup_frame = mark_search_strategy_windows(setup_frame)
    else:
        setup_frame["is_search_setup"] = 0

    setup_frame = setup_frame[["ts_code", "trade_date", "is_search_setup"]]
    prediction_frame = feature_frame.merge(setup_frame, on=["ts_code", "trade_date"], how="left")
    prediction_frame["is_search_setup"] = prediction_frame["is_search_setup"].fillna(0).astype("int8")
    prediction_frame = prediction_frame.dropna(subset=feature_columns).reset_index(drop=True)

    setup_rows = int((prediction_frame["is_search_setup"] == 1).sum())
    print(
        f"[feature] prediction rows: {len(prediction_frame)}, "
        f"search setup rows: {setup_rows}"
    )

    training_frame = prediction_frame[prediction_frame["is_search_setup"] == 1].dropna(
        subset=["breakout_label", "upside_5d"]
    ).reset_index(drop=True)
    print(f"[feature] labeled search setup rows: {len(training_frame)}")

    return training_frame, prediction_frame, feature_columns


def split_frames(training_frame):
    gap = pd.Timedelta(days=SPLIT_GAP_DAYS)
    train_end = pd.Timestamp(TRAIN_END_DATE) - gap
    valid_start = pd.Timestamp(TRAIN_END_DATE) + gap
    valid_end = pd.Timestamp(VALID_END_DATE) - gap
    test_start = pd.Timestamp(VALID_END_DATE) + gap
    test_end = pd.Timestamp(TEST_END_DATE)

    train_frame = training_frame[training_frame["trade_date"] <= train_end].copy()
    valid_frame = training_frame[
        (training_frame["trade_date"] >= valid_start) & (training_frame["trade_date"] <= valid_end)
    ].copy()
    test_frame = training_frame[
        (training_frame["trade_date"] >= test_start) & (training_frame["trade_date"] <= test_end)
    ].copy()

    if train_frame.empty or valid_frame.empty or test_frame.empty:
        raise ValueError("时间切分后存在空数据集，请检查日期范围或缓存数据")

    return train_frame, valid_frame, test_frame


def maybe_sample_train_frame(train_frame, max_rows=MAX_TRAIN_ROWS):
    if len(train_frame) <= max_rows:
        return train_frame
    return train_frame.sample(n=max_rows, random_state=RANDOM_STATE).sort_values("trade_date").reset_index(drop=True)


def make_lgbm_model(alpha, use_gpu=USE_GPU):
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    if use_gpu:
        params.update(
            {
                "device_type": GPU_DEVICE_TYPE,
                "gpu_platform_id": GPU_PLATFORM_ID,
                "gpu_device_id": GPU_DEVICE_ID,
                "gpu_use_dp": GPU_USE_DP,
                "max_bin": 63,
            }
        )

    return lgb.LGBMRegressor(**params)


def make_classifier_model(use_gpu=USE_GPU):
    params = {
        "objective": "binary",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if use_gpu:
        params.update(
            {
                "device_type": GPU_DEVICE_TYPE,
                "gpu_platform_id": GPU_PLATFORM_ID,
                "gpu_device_id": GPU_DEVICE_ID,
                "gpu_use_dp": GPU_USE_DP,
                "max_bin": 63,
            }
        )
    return lgb.LGBMClassifier(**params)


def fit_lgbm_model(model, x_train, y_train, x_valid, y_valid):
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="quantile",
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    return model


def fit_classifier(train_frame, valid_frame, feature_columns):
    x_train = train_frame[feature_columns]
    y_train = train_frame["breakout_label"]
    x_valid = valid_frame[feature_columns]
    y_valid = valid_frame["breakout_label"]

    model = make_classifier_model()
    try:
        print(f"[train] breakout classifier device={'gpu' if USE_GPU else 'cpu'}")
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
    except lgb.basic.LightGBMError as error:
        if not USE_GPU or not FALLBACK_TO_CPU_IF_GPU_FAILS:
            raise
        print(f"[train] GPU training failed: {error}")
        print("[train] fallback to CPU for classifier")
        model = make_classifier_model(use_gpu=False)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
    valid_pred = model.predict_proba(x_valid, num_iteration=model.best_iteration_)[:, 1]
    return model, valid_pred


def fit_horizon_models(train_frame, valid_frame, feature_columns):
    x_train = train_frame[feature_columns]
    y_train = train_frame["upside_5d"]
    x_valid = valid_frame[feature_columns]
    y_valid = valid_frame["upside_5d"]

    trained_models = {}
    valid_predictions = {}

    for alpha in QUANTILES:
        model = make_lgbm_model(alpha)
        try:
            print(
                f"[train] upside_5d alpha={alpha} "
                f"device={'gpu' if USE_GPU else 'cpu'}"
            )
            model = fit_lgbm_model(model, x_train, y_train, x_valid, y_valid)
        except lgb.basic.LightGBMError as error:
            if not USE_GPU or not FALLBACK_TO_CPU_IF_GPU_FAILS:
                raise

            print(f"[train] GPU training failed: {error}")
            print("[train] fallback to CPU for this model")
            model = make_lgbm_model(alpha, use_gpu=False)
            model = fit_lgbm_model(model, x_train, y_train, x_valid, y_valid)

        trained_models[alpha] = model
        valid_predictions[alpha] = model.predict(x_valid, num_iteration=model.best_iteration_)
        completed = 1 + list(QUANTILES).index(alpha) + 1
        show_progress(completed, 4, "model 训练模型")

    width = valid_predictions[0.9] - valid_predictions[0.1]
    sign_hit = (np.sign(valid_predictions[0.5]) == np.sign(y_valid.to_numpy())).astype(float)

    calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, increasing=False, out_of_bounds="clip")
    calibrator.fit(width, sign_hit)

    return trained_models, calibrator


def train_model_bundle(train_frame, valid_frame, feature_columns):
    bundle = {
        "feature_columns": feature_columns,
        "quantiles": list(QUANTILES),
        "models": {},
        "classifier": None,
        "confidence_calibrators": {},
    }

    show_progress(0, 4, "model 训练模型")
    classifier, valid_breakout_prob = fit_classifier(train_frame, valid_frame, feature_columns)
    show_progress(1, 4, "model 训练模型")
    bundle["classifier"] = classifier
    bundle["valid_breakout_prob"] = valid_breakout_prob

    models, calibrator = fit_horizon_models(train_frame, valid_frame, feature_columns)
    bundle["models"]["upside_5d"] = models
    bundle["confidence_calibrators"]["upside_5d"] = calibrator
    show_progress(4, 4, "model 训练模型")

    return bundle


def predict_with_bundle(bundle, frame):
    predictions = frame[["ts_code", "trade_date"]].copy()
    x = frame[bundle["feature_columns"]]

    breakout_prob = bundle["classifier"].predict_proba(x, num_iteration=bundle["classifier"].best_iteration_)[:, 1]
    predictions["breakout_prob"] = breakout_prob

    upside_models = bundle["models"]["upside_5d"]
    p10 = upside_models[0.1].predict(x, num_iteration=upside_models[0.1].best_iteration_)
    p50 = upside_models[0.5].predict(x, num_iteration=upside_models[0.5].best_iteration_)
    p90 = upside_models[0.9].predict(x, num_iteration=upside_models[0.9].best_iteration_)
    width = p90 - p10
    confidence = bundle["confidence_calibrators"]["upside_5d"].predict(width)

    predictions["upside_p10"] = p10
    predictions["upside_p50"] = p50
    predictions["upside_p90"] = p90
    predictions["upside_conf"] = confidence
    predictions["score"] = predictions["breakout_prob"] * np.maximum(predictions["upside_p50"], 0)

    return predictions


def evaluate_bundle(bundle, test_frame):
    pred_frame = predict_with_bundle(bundle, test_frame)
    breakout_actual = test_frame["breakout_label"].to_numpy()
    if len(np.unique(breakout_actual)) > 1:
        breakout_auc = float(roc_auc_score(breakout_actual, pred_frame["breakout_prob"]))
    else:
        breakout_auc = None

    metrics = {
        "breakout": {
            "auc": breakout_auc,
            "mean_prob": float(pred_frame["breakout_prob"].mean()),
            "positive_rate": float(np.mean(breakout_actual)),
        }
    }

    actual = test_frame["upside_5d"].to_numpy()
    p10 = pred_frame["upside_p10"].to_numpy()
    p50 = pred_frame["upside_p50"].to_numpy()
    p90 = pred_frame["upside_p90"].to_numpy()
    metrics["upside_5d"] = {
        "mae": float(mean_absolute_error(actual, p50)),
        "rmse": float(mean_squared_error(actual, p50) ** 0.5),
        "interval_80_coverage": float(np.mean((actual >= p10) & (actual <= p90))),
        "mean_interval_width": float(np.mean(p90 - p10)),
        "mean_confidence": float(np.mean(pred_frame["upside_conf"].to_numpy())),
    }

    return metrics, pred_frame


def save_bundle(bundle, metrics):
    with open(MODEL_BUNDLE_FILE, "wb") as file:
        pickle.dump(bundle, file)

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def build_latest_predictions(bundle, prediction_frame):
    latest_rows = (
        prediction_frame.sort_values(["ts_code", "trade_date"])
        .groupby("ts_code", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest_pred = predict_with_bundle(bundle, latest_rows)
    latest_pred = latest_pred.sort_values("score", ascending=False).reset_index(drop=True)
    latest_pred.to_csv(LATEST_PRED_FILE, index=False, encoding="utf-8-sig")
    return latest_pred


def print_metrics(metrics):
    print("\n[test metrics]")
    breakout = metrics["breakout"]
    upside = metrics["upside_5d"]
    auc_text = "NA" if breakout["auc"] is None else f"{breakout['auc']:.4f}"
    print(
        f"breakout: AUC={auc_text}, mean_prob={breakout['mean_prob']:.4f}, "
        f"positive_rate={breakout['positive_rate']:.4f}"
    )
    print(
        f"upside_5d: MAE={upside['mae']:.5f}, RMSE={upside['rmse']:.5f}, "
        f"COVERAGE={upside['interval_80_coverage']:.4f}, CONF={upside['mean_confidence']:.4f}"
    )


def main():
    ensure_dirs()
    print("[stage] 1/7 加载股票池")
    stock_codes = load_stock_universe()
    print(f"[setup] stock universe: {len(stock_codes)}")

    print("[stage] 2/7 加载或下载训练行情")
    pro = create_pro()
    panel = load_or_download_panel(pro, stock_codes)
    print(f"[data] panel rows: {len(panel)}")

    print("[stage] 3/7 构造滚动样本和特征")
    training_frame, prediction_frame, feature_columns = build_feature_frames(panel)
    print(f"[feature] training rows: {len(training_frame)}, features: {len(feature_columns)}")

    print("[stage] 4/7 按时间切分训练/验证/测试集")
    train_frame, valid_frame, test_frame = split_frames(training_frame)
    train_frame = maybe_sample_train_frame(train_frame)
    print(
        f"[split] train={len(train_frame)}, valid={len(valid_frame)}, test={len(test_frame)}"
    )

    print("[stage] 5/7 训练突破概率和上涨空间模型")
    bundle = train_model_bundle(train_frame, valid_frame, feature_columns)
    print("[stage] 6/7 测试集评估")
    metrics, _ = evaluate_bundle(bundle, test_frame)
    print("[stage] 7/7 保存模型和最新预测")
    save_bundle(bundle, metrics)
    latest_pred = build_latest_predictions(bundle, prediction_frame)

    print_metrics(metrics)
    print(f"\n[save] model bundle: {MODEL_BUNDLE_FILE}")
    print(f"[save] metrics: {METRICS_FILE}")
    print(f"[save] latest predictions: {LATEST_PRED_FILE}")
    print("\n[top 10 latest]")
    print(latest_pred.head(10))


if __name__ == "__main__":
    main()
