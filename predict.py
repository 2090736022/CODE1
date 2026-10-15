import argparse
import pickle
from pathlib import Path

import pandas as pd

import config
from stock_filters import filter_non_st_codes
from model import (
    LATEST_PRED_FILE,
    MODEL_BUNDLE_FILE,
    PANEL_CACHE_FILE,
    START_DATE,
    build_feature_frames,
    create_pro,
    ensure_dirs,
    fetch_trade_date_panel,
    get_trade_dates,
    is_limit_up_row,
    load_stock_universe,
    predict_with_bundle,
    row_matches_search_strategy,
)


BASE_DIR = Path(__file__).resolve().parent
PREDICT_DIR = BASE_DIR / "model_artifacts"
DOWNLOAD_DAILY_BASIC_DIR = BASE_DIR / "model_artifacts" / "download_daily" / "daily_basic"
DOWNLOAD_DAILY_DIR = BASE_DIR / "model_artifacts" / "download_daily" / "daily"
TOP_N = 10
BACKTEST_TOP_N = 3
BACKTEST_FORWARD_DAYS = config.BACKTEST_FORWARD_DAYS
LATEST_DOWN_RET_THRESHOLD = 0.0


def get_predict_date(predict_date=None):
    predict_date = str(predict_date or config.get_end_date()).strip()
    if not (predict_date.isdigit() and len(predict_date) == 8):
        raise ValueError(f"config.py 中的 END_DATE/DAILY_RUN_DATE 日期格式不正确: {predict_date}")
    return predict_date


def load_bundle(bundle_file=MODEL_BUNDLE_FILE):
    bundle_file = Path(bundle_file)
    if not bundle_file.exists():
        raise FileNotFoundError(f"未找到训练好的模型文件: {bundle_file}")

    with open(bundle_file, "rb") as file:
        return pickle.load(file)


def load_cached_panel(panel_file=PANEL_CACHE_FILE):
    panel_file = Path(panel_file)
    if not panel_file.exists():
        return pd.DataFrame()
    return pd.read_csv(panel_file, dtype={"ts_code": str, "trade_date": str})


def repair_panel_with_download_cache(panel):
    if panel.empty:
        return panel

    panel = panel.copy()
    panel["trade_date"] = panel["trade_date"].astype(str)
    need_basic = panel["turnover_rate"].isna() | panel["total_mv"].isna()
    dates = sorted(panel.loc[need_basic, "trade_date"].dropna().astype(str).unique().tolist())
    repaired_dates = []

    for trade_date in dates:
        basic_file = DOWNLOAD_DAILY_BASIC_DIR / f"{trade_date}.csv"
        if not basic_file.exists():
            continue

        basic = pd.read_csv(basic_file, dtype={"ts_code": str, "trade_date": str})
        if basic.empty or not {"ts_code", "turnover_rate", "total_mv"}.issubset(basic.columns):
            continue

        basic = basic[["ts_code", "turnover_rate", "total_mv"]].copy()
        day_mask = panel["trade_date"] == trade_date
        day = panel.loc[day_mask].drop(columns=["turnover_rate", "total_mv"]).merge(
            basic,
            on="ts_code",
            how="left",
        )
        panel.loc[day_mask, "turnover_rate"] = day["turnover_rate"].to_numpy()
        panel.loc[day_mask, "total_mv"] = day["total_mv"].to_numpy()
        repaired_dates.append(trade_date)

    if repaired_dates:
        print(f"[predict-data] repaired daily_basic for dates: {', '.join(repaired_dates[-5:])}")

    return panel


def refresh_panel_to_date(pro, stock_codes, as_of_date, panel_file=PANEL_CACHE_FILE):
    panel_file = Path(panel_file)
    cached_panel = load_cached_panel(panel_file)

    if cached_panel.empty:
        start_date = START_DATE
    else:
        cached_panel = cached_panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        last_cached_date = str(cached_panel["trade_date"].max())
        if last_cached_date >= as_of_date:
            cached_panel = repair_panel_with_download_cache(cached_panel)
            return cached_panel[cached_panel["ts_code"].isin(stock_codes)].copy()

        start_date = (
            pd.to_datetime(last_cached_date) + pd.Timedelta(days=1)
        ).strftime("%Y%m%d")

    trade_dates = get_trade_dates(pro, start_date=start_date, end_date=as_of_date)
    if not trade_dates:
        cached_panel = repair_panel_with_download_cache(cached_panel)
        return cached_panel[cached_panel["ts_code"].isin(stock_codes)].copy()

    allowed_codes = set(stock_codes)
    new_frames = []
    total_days = len(trade_dates)

    for index, trade_date in enumerate(trade_dates, start=1):
        day_df = fetch_trade_date_panel(pro, trade_date, allowed_codes)
        if not day_df.empty:
            new_frames.append(day_df)

        if index == 1 or index % 20 == 0 or index == total_days:
            print(f"[predict-data] {index}/{total_days} trade dates refreshed")

    if new_frames:
        new_panel = pd.concat(new_frames, ignore_index=True)
        panel = pd.concat([cached_panel, new_panel], ignore_index=True)
        panel = panel.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
        panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        panel = panel[panel["ts_code"].isin(stock_codes)].copy()
        panel = repair_panel_with_download_cache(panel)
        panel.to_csv(panel_file, index=False, encoding="utf-8-sig")
        return panel

    cached_panel = repair_panel_with_download_cache(cached_panel)
    return cached_panel[cached_panel["ts_code"].isin(stock_codes)].copy()


def build_latest_rows(prediction_frame):
    latest_rows = (
        prediction_frame.sort_values(["ts_code", "trade_date"])
        .groupby("ts_code", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest_rows


def build_as_of_rows(prediction_frame, as_of_date):
    as_of_timestamp = pd.to_datetime(as_of_date)
    as_of_frame = prediction_frame[prediction_frame["trade_date"] <= as_of_timestamp].copy()
    if as_of_frame.empty:
        raise ValueError(f"{as_of_date} 之前没有可用于预测的特征数据")
    return build_latest_rows(as_of_frame)


def mark_latest_search_setup_only(panel, latest_rows):
    setup_map = {}
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    for ts_code, group_df in panel.groupby("ts_code", group_keys=False):
        setup_map[ts_code] = int(row_matches_search_strategy(group_df))

    latest_rows = latest_rows.copy()
    latest_rows["is_search_setup"] = latest_rows["ts_code"].map(setup_map).fillna(0).astype("int8")
    return latest_rows


def mark_as_of_search_setup_only(panel, latest_rows):
    setup_map = {}
    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    for _, row in latest_rows[["ts_code", "trade_date"]].iterrows():
        ts_code = row["ts_code"]
        trade_date = row["trade_date"]
        history = panel[(panel["ts_code"] == ts_code) & (panel["trade_date"] <= trade_date)]
        setup_map[ts_code] = int(row_matches_search_strategy(history))

    latest_rows = latest_rows.copy()
    latest_rows["is_search_setup"] = latest_rows["ts_code"].map(setup_map).fillna(0).astype("int8")
    return latest_rows


def attach_stock_meta(pred_frame, stock_file=None):
    stock_file = Path(stock_file or config.STOCK_FILE)
    stock_meta = pd.read_csv(stock_file, dtype={"ts_code": str})

    keep_columns = [column for column in ["ts_code", "symbol", "name", "area", "industry"] if column in stock_meta.columns]
    if "ts_code" not in keep_columns:
        return pred_frame

    stock_meta = stock_meta[keep_columns].drop_duplicates(subset=["ts_code"])
    merged = pred_frame.merge(stock_meta, on="ts_code", how="left")

    ordered_columns = ["ts_code", "symbol", "name", "area", "industry", "trade_date"]
    other_columns = [column for column in merged.columns if column not in ordered_columns]
    return merged[[column for column in ordered_columns if column in merged.columns] + other_columns]


def save_prediction_output(pred_frame, as_of_date, output_dir=PREDICT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    dated_file = output_dir / f"predictions_{as_of_date}.csv"
    pred_frame.to_csv(dated_file, index=False, encoding="utf-8-sig")
    pred_frame.to_csv(LATEST_PRED_FILE, index=False, encoding="utf-8-sig")
    return dated_file


def cache_code(ts_code):
    return str(ts_code).replace(".", "_")


def read_cached_daily(ts_code):
    cache_file = DOWNLOAD_DAILY_DIR / f"{cache_code(ts_code)}.csv"
    if not cache_file.exists():
        return None

    return pd.read_csv(cache_file, dtype={"ts_code": str, "trade_date": str})


def calculate_forward_5d_return(ts_code, trade_date):
    daily_df = read_cached_daily(ts_code)
    if daily_df is None or daily_df.empty:
        return None

    daily_df = daily_df.copy()
    daily_df["trade_date"] = daily_df["trade_date"].astype(str)
    daily_df["open"] = pd.to_numeric(daily_df["open"], errors="coerce")
    daily_df["close"] = pd.to_numeric(daily_df["close"], errors="coerce")
    daily_df = daily_df.drop_duplicates(subset=["trade_date"], keep="last")
    daily_df = daily_df.sort_values("trade_date").reset_index(drop=True)

    trade_date = pd.to_datetime(trade_date).strftime("%Y%m%d")
    matched_index = daily_df.index[daily_df["trade_date"] == trade_date]
    if len(matched_index) == 0:
        return None

    selected_index = matched_index[0]
    future_df = daily_df.iloc[selected_index + 1 : selected_index + 1 + BACKTEST_FORWARD_DAYS].copy()
    if len(future_df) < BACKTEST_FORWARD_DAYS:
        return None

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
        if best_result is None or return_ratio > best_result["future_5d"]:
            best_result = {
                "future_5d": return_ratio,
            }

    if best_result is None:
        return None

    return {
        **best_result,
    }


def attach_backtest_result(pred_frame):
    backtest_columns = ["future_5d"]
    if pred_frame is None or pred_frame.empty:
        return pd.DataFrame(columns=list(pred_frame.columns) + backtest_columns)

    rows = []
    for _, row in pred_frame.iterrows():
        backtest = calculate_forward_5d_return(row["ts_code"], row["trade_date"])
        if backtest is None:
            continue

        output_row = row.to_dict()
        output_row.update(backtest)
        rows.append(output_row)

    if not rows:
        return pd.DataFrame(columns=list(pred_frame.columns) + backtest_columns)

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run latest stock prediction.")
    parser.add_argument(
        "ts_code",
        nargs="?",
        help="Optional single stock code, for example 000001.SZ.",
    )
    parser.add_argument(
        "-d",
        "--date",
        dest="predict_date",
        help="Optional prediction date in YYYYMMDD format. Defaults to config.get_end_date().",
    )
    return parser.parse_args()


def normalize_ts_code(ts_code, stock_codes):
    if not ts_code:
        return None

    ts_code = str(ts_code).strip().upper()
    if ts_code in stock_codes:
        return ts_code

    if ts_code.isdigit() and len(ts_code) == 6:
        matches = [code for code in stock_codes if code.startswith(ts_code + ".")]
        if len(matches) == 1:
            return matches[0]

    raise ValueError(f"股票代码不在当前股票池中: {ts_code}")


def build_rank_output(pred_frame, stock_code=None, top_n=TOP_N):
    output = pred_frame.sort_values("score", ascending=False).reset_index(drop=True)
    output["rank"] = output.index + 1
    output = output[["ts_code", "trade_date", "score", "rank"]]

    if stock_code:
        output = output[output["ts_code"] == stock_code].reset_index(drop=True)
        if output.empty:
            raise ValueError(f"{stock_code} 当前预测日期没有通过筛选条件，无法输出预测排名")
        return output

    return output.head(top_n).reset_index(drop=True)


def filter_prev_day_not_limit_up(panel, latest_rows):
    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    blocked_codes = set()
    for _, row in latest_rows[["ts_code", "trade_date"]].iterrows():
        ts_code = row["ts_code"]
        trade_date = pd.to_datetime(row["trade_date"])
        history = panel[(panel["ts_code"] == ts_code) & (panel["trade_date"] <= trade_date)]
        if len(history) < 2:
            continue

        prev_day = history.iloc[-2]
        if is_limit_up_row(prev_day):
            blocked_codes.add(ts_code)

    if not blocked_codes:
        return latest_rows

    return latest_rows[~latest_rows["ts_code"].isin(blocked_codes)].reset_index(drop=True)


def run_predict(stock_code=None, predict_date=None):
    ensure_dirs()
    as_of_date = get_predict_date(predict_date)

    bundle = load_bundle()
    stock_codes = load_stock_universe()
    stock_codes = filter_non_st_codes(stock_codes, config.STOCK_FILE)
    stock_code = normalize_ts_code(stock_code, stock_codes)
    print(f"[predict] stock universe: {len(stock_codes)}")
    print(f"[predict] as_of_date: {as_of_date}")
    if stock_code:
        print(f"[predict] single stock: {stock_code}")

    pro = create_pro()
    panel = refresh_panel_to_date(pro, stock_codes, as_of_date)
    if panel.empty:
        raise ValueError("没有可用于预测的行情数据")

    _, prediction_frame, _ = build_feature_frames(panel, mark_all_search_windows=False)
    latest_rows = build_as_of_rows(prediction_frame, as_of_date)
    actual_predict_date = latest_rows["trade_date"].max()
    print(f"[predict] actual_predict_date: {actual_predict_date:%Y%m%d}")
    latest_rows = mark_as_of_search_setup_only(panel, latest_rows)
    latest_rows = latest_rows[latest_rows["is_search_setup"] == 1].reset_index(drop=True)
    if latest_rows.empty:
        raise ValueError("当前预测日期没有符合 lianghua1_search 判定条件的股票")

    latest_rows = latest_rows[latest_rows["daily_ret"] < LATEST_DOWN_RET_THRESHOLD].reset_index(drop=True)
    if latest_rows.empty:
        raise ValueError("当前预测日期没有当日下跌且符合条件的股票")

    latest_rows = filter_prev_day_not_limit_up(panel, latest_rows)
    if latest_rows.empty:
        raise ValueError("当前预测日期没有倒数第二日未涨停且符合条件的股票")

    pred_frame = predict_with_bundle(bundle, latest_rows)
    extra_columns = ["is_search_setup", "daily_ret"]
    pred_frame = pred_frame.merge(
        latest_rows[["ts_code", "trade_date"] + [column for column in extra_columns if column in latest_rows.columns]],
        on=["ts_code", "trade_date"],
        how="left",
    )
    pred_frame = build_rank_output(pred_frame, stock_code=stock_code)
    if not stock_code:
        pred_frame = pred_frame.head(BACKTEST_TOP_N).reset_index(drop=True)
    pred_frame = attach_backtest_result(pred_frame)
    print(f"[predict] filter latest daily_ret < {LATEST_DOWN_RET_THRESHOLD:.2%}, rows={len(pred_frame)}")

    output_file = save_prediction_output(pred_frame, as_of_date)
    print(f"[predict] saved latest predictions: {output_file}")
    print(pred_frame)
    return pred_frame


def main():
    args = parse_args()
    run_predict(args.ts_code, args.predict_date)


if __name__ == "__main__":
    main()
