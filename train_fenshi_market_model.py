# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import time
from pathlib import Path

import akshare as ak
import pandas as pd

from fenshi import format_em_symbol
from fenshi_strength import (
    FEATURE_COLUMNS,
    MARKET_MODEL_PATH,
    calculate_day_features,
    get_available_dates,
    prepare_minute_data,
)


BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "model_artifacts" / "download_daily"
DAILY_CACHE_DIR = DOWNLOAD_DIR / "daily"
DAILY_BASIC_CACHE_DIR = DOWNLOAD_DIR / "daily_basic"
DEFAULT_STOCK_FILE = BASE_DIR / "number.csv"
TRAIN_SAMPLE_PATH = BASE_DIR / "model_artifacts" / "fenshi_market_training_samples.csv"


def normalize_yyyymmdd(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip().replace("-", "")
    if len(value) != 8 or not value.isdigit():
        raise ValueError("日期格式请使用 YYYYMMDD 或 YYYY-MM-DD")
    return value


def yyyymmdd_to_iso(value: str) -> str:
    value = normalize_yyyymmdd(value)
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def iso_to_yyyymmdd(value: str) -> str:
    return str(value).replace("-", "")


def cache_code(ts_code: str) -> str:
    return str(ts_code).strip().replace(".", "_")


def ts_code_to_em_symbol(ts_code: str) -> str:
    code = str(ts_code).strip()
    if "." in code:
        number, market = code.split(".", 1)
        market = market.lower()
        if market == "sh":
            return f"sh{number}"
        if market == "sz":
            return f"sz{number}"
    return format_em_symbol(code)


def read_csv(cache_file: Path) -> pd.DataFrame:
    if not cache_file.exists():
        return pd.DataFrame()
    return pd.read_csv(cache_file, dtype={"ts_code": str, "trade_date": str})


def load_stock_codes(stock_file: Path) -> list[str]:
    frame = pd.read_csv(stock_file, dtype={"ts_code": str})
    if "ts_code" not in frame.columns:
        raise ValueError(f"{stock_file} 缺少 ts_code 列")
    return (
        frame["ts_code"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )


def load_qualified_daily_basic(
    daily_basic_dir: Path,
    start_date: str | None,
    end_date: str | None,
    min_turnover_rate: float,
    min_total_mv: float,
) -> pd.DataFrame:
    frames = []
    for cache_file in sorted(daily_basic_dir.glob("*.csv")):
        trade_date = cache_file.stem
        if start_date and trade_date < start_date:
            continue
        if end_date and trade_date > end_date:
            continue

        frame = read_csv(cache_file)
        if frame.empty:
            continue
        required = {"ts_code", "trade_date", "turnover_rate", "total_mv"}
        if not required.issubset(frame.columns):
            continue

        frame["turnover_rate"] = pd.to_numeric(frame["turnover_rate"], errors="coerce")
        frame["total_mv"] = pd.to_numeric(frame["total_mv"], errors="coerce")
        frame = frame[
            (frame["turnover_rate"] > min_turnover_rate)
            & (frame["total_mv"] > min_total_mv)
        ].copy()
        if not frame.empty:
            frames.append(frame[["ts_code", "trade_date", "turnover_rate", "total_mv"]])

    if not frames:
        return pd.DataFrame(columns=["ts_code", "trade_date", "turnover_rate", "total_mv"])

    result = pd.concat(frames, ignore_index=True)
    result["ts_code"] = result["ts_code"].astype(str).str.strip()
    result["trade_date"] = result["trade_date"].astype(str).str.strip()
    return result.drop_duplicates(subset=["ts_code", "trade_date"])


def load_daily_cache(daily_dir: Path, ts_code: str) -> pd.DataFrame:
    frame = read_csv(daily_dir / f"{cache_code(ts_code)}.csv")
    if frame.empty or "trade_date" not in frame.columns:
        return pd.DataFrame()

    for column in ("open", "close", "high", "low", "vol", "amount", "pct_chg"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trade_date"] = frame["trade_date"].astype(str).str.strip()
    frame = frame.drop_duplicates(subset=["trade_date"], keep="last")
    return frame.sort_values("trade_date").reset_index(drop=True)


def fetch_minute_data(ts_code: str) -> pd.DataFrame:
    return ak.stock_zh_a_minute(
        symbol=ts_code_to_em_symbol(ts_code),
        period="1",
        adjust="",
    )


def build_stock_samples(
    ts_code: str,
    qualified: pd.DataFrame,
    daily: pd.DataFrame,
    recent_window: int,
    label_return: float,
    include_unmatched_dates: bool,
) -> list[dict[str, float | int | str]]:
    raw_minute = fetch_minute_data(ts_code)
    minute_data, columns = prepare_minute_data(raw_minute)
    minute_dates = get_available_dates(minute_data)
    if not minute_dates:
        return []

    qualified_by_date = qualified.set_index("trade_date", drop=False)
    daily_by_date = daily.set_index("trade_date", drop=False) if not daily.empty else pd.DataFrame()
    qualified_dates = set(qualified["trade_date"].astype(str))
    daily_dates = set(daily["trade_date"].astype(str)) if not daily.empty else set()

    matched_iso_dates = [
        date for date in minute_dates
        if iso_to_yyyymmdd(date) in qualified_dates
    ]
    if not matched_iso_dates and include_unmatched_dates:
        matched_iso_dates = minute_dates

    rows = []
    for trade_date_iso in matched_iso_dates:
        trade_date = iso_to_yyyymmdd(trade_date_iso)
        day_data = minute_data[minute_data["_trade_date"] == trade_date_iso]
        if len(day_data) < max(20, recent_window):
            continue

        features = calculate_day_features(day_data, columns, recent_window)
        if trade_date in daily_dates:
            daily_row = daily_by_date.loc[trade_date]
            open_price = float(daily_row["open"])
            close_price = float(daily_row["close"])
            label_value = (close_price / open_price - 1.0) if open_price > 0 else features["day_return"]
            label_source = "daily"
        else:
            label_value = features["day_return"]
            label_source = "minute"

        basic_row = qualified_by_date.loc[trade_date] if trade_date in qualified_dates else None
        sample = {
            **features,
            "ts_code": ts_code,
            "trade_date": trade_date,
            "label": int(label_value >= label_return),
            "label_return": float(label_value),
            "label_source": label_source,
            "matched_daily_basic": int(trade_date in qualified_dates),
            "matched_daily_k": int(trade_date in daily_dates),
            "turnover_rate": float(basic_row["turnover_rate"]) if basic_row is not None else pd.NA,
            "total_mv": float(basic_row["total_mv"]) if basic_row is not None else pd.NA,
        }
        rows.append(sample)

    return rows


def train_market_model(
    samples: pd.DataFrame,
    output_model: Path,
    label_return: float,
    recent_window: int,
    min_samples: int,
) -> None:
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("训练模型需要 scikit-learn 和 joblib，请先安装依赖") from exc

    frame = samples.dropna(subset=list(FEATURE_COLUMNS) + ["label"]).copy()
    if len(frame) < min_samples:
        raise ValueError(f"可训练样本太少: {len(frame)}，低于 min_samples={min_samples}")
    if frame["label"].nunique() < 2:
        raise ValueError("训练标签只有单一类别，请调整 --label-return 或扩大训练日期范围")
    if frame["label"].value_counts().min() < 2:
        raise ValueError("正负样本中至少有一类少于 2 条，请调整 --label-return 或扩大训练日期范围")

    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.25,
        random_state=42,
        stratify=frame["label"],
    )
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(train_frame[list(FEATURE_COLUMNS)], train_frame["label"])

    prediction = pipeline.predict(test_frame[list(FEATURE_COLUMNS)])
    probability = pipeline.predict_proba(test_frame[list(FEATURE_COLUMNS)])[:, 1]
    accuracy = accuracy_score(test_frame["label"], prediction)
    auc = roc_auc_score(test_frame["label"], probability)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "recent_window": recent_window,
            "label_return": label_return,
            "sample_count": len(frame),
            "scope": "market",
            "filters": {
                "source": "turnover_rate > threshold and total_mv > threshold",
            },
        },
        output_model,
    )

    print(f"模型已保存: {output_model}")
    print(f"训练样本: {len(train_frame)}，测试样本: {len(test_frame)}")
    print(f"正样本占比: {frame['label'].mean():.2%}")
    print(f"测试准确率: {accuracy:.2%}")
    print(f"测试 AUC: {auc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练全市场分时成交强度模型")
    parser.add_argument("--daily-dir", type=Path, default=DAILY_CACHE_DIR, help="日K缓存目录")
    parser.add_argument("--daily-basic-dir", type=Path, default=DAILY_BASIC_CACHE_DIR, help="daily_basic缓存目录")
    parser.add_argument("--stock-file", type=Path, default=DEFAULT_STOCK_FILE, help="股票池文件，默认 number.csv")
    parser.add_argument("--start-date", help="训练起始日期，YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--end-date", help="训练结束日期，YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--min-turnover-rate", type=float, default=3.0, help="最小换手率，默认 3")
    parser.add_argument("--min-total-mv", type=float, default=400000.0, help="最小总市值，单位万元，默认 400000 即 40 亿")
    parser.add_argument("--label-return", type=float, default=0.005, help="强势标签阈值，默认 0.005 表示 0.5%%")
    parser.add_argument("--recent-window", type=int, default=30, help="特征里的短线窗口分钟数，默认 30")
    parser.add_argument("--request-sleep", type=float, default=0.2, help="拉取分时数据后的等待秒数")
    parser.add_argument("--max-stocks", type=int, help="最多训练多少只股票，调试时可设置")
    parser.add_argument("--min-samples", type=int, default=50, help="最小训练样本数")
    parser.add_argument("--output-model", type=Path, default=MARKET_MODEL_PATH, help="模型保存路径")
    parser.add_argument("--sample-output", type=Path, default=TRAIN_SAMPLE_PATH, help="训练样本明细保存路径")
    parser.add_argument(
        "--include-unmatched-dates",
        action="store_true",
        help="开启兜底：分时日期无法匹配合格daily_basic时，也纳入该股票拉到的分时日期",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = normalize_yyyymmdd(args.start_date)
    end_date = normalize_yyyymmdd(args.end_date)

    qualified = load_qualified_daily_basic(
        daily_basic_dir=args.daily_basic_dir,
        start_date=start_date,
        end_date=end_date,
        min_turnover_rate=args.min_turnover_rate,
        min_total_mv=args.min_total_mv,
    )
    if qualified.empty:
        raise ValueError("没有找到符合换手率和市值条件的daily_basic数据")

    stock_pool = set(load_stock_codes(args.stock_file))
    qualified = qualified[qualified["ts_code"].isin(stock_pool)].copy()
    if qualified.empty:
        raise ValueError("number.csv 股票池中没有找到同时满足换手率和市值条件的记录")

    stock_codes = sorted(qualified["ts_code"].dropna().astype(str).unique().tolist())
    if args.max_stocks:
        stock_codes = stock_codes[: args.max_stocks]

    print(f"符合条件的股票数: {len(stock_codes)}")
    print(f"符合条件的股票-日期样本数: {len(qualified)}")
    print(f"日期范围: {start_date or '全部'} - {end_date or '全部'}")

    all_rows = []
    for index, ts_code in enumerate(stock_codes, start=1):
        stock_qualified = qualified[qualified["ts_code"] == ts_code].copy()
        daily = load_daily_cache(args.daily_dir, ts_code)
        try:
            rows = build_stock_samples(
                ts_code=ts_code,
                qualified=stock_qualified,
                daily=daily,
                recent_window=args.recent_window,
                label_return=args.label_return,
                include_unmatched_dates=args.include_unmatched_dates,
            )
            all_rows.extend(rows)
            print(f"[{index}/{len(stock_codes)}] {ts_code} 样本 {len(rows)}")
        except Exception as exc:
            print(f"[{index}/{len(stock_codes)}] {ts_code} 跳过: {exc}")
        finally:
            if args.request_sleep > 0:
                time.sleep(args.request_sleep)

    samples = pd.DataFrame(all_rows)
    if samples.empty:
        raise ValueError("没有生成任何训练样本，请检查分时数据是否可拉取")
    if not args.include_unmatched_dates:
        samples["turnover_rate"] = pd.to_numeric(samples["turnover_rate"], errors="coerce")
        samples["total_mv"] = pd.to_numeric(samples["total_mv"], errors="coerce")
        samples = samples[
            (samples["matched_daily_basic"] == 1)
            & (samples["turnover_rate"] > args.min_turnover_rate)
            & (samples["total_mv"] > args.min_total_mv)
        ].copy()
        if samples.empty:
            raise ValueError("严格过滤后没有训练样本，请检查分时日期和daily_basic日期是否对应")

    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.sample_output, index=False, encoding="utf-8-sig")
    print(f"训练样本明细已保存: {args.sample_output}")

    train_market_model(
        samples=samples,
        output_model=args.output_model,
        label_return=args.label_return,
        recent_window=args.recent_window,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()
