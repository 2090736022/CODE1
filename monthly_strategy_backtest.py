import argparse
import contextlib
import io
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
import lianghua1_search_botton
import lianghua1_search_test
import lianghua1_search_turn


BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "model_artifacts" / "download_daily"
DAILY_BASIC_DIR = DOWNLOAD_DIR / "daily_basic"
OUTPUT_DIR = BASE_DIR / "model_artifacts" / "monthly_backtest"
BACKTEST_FORWARD_DAYS = config.BACKTEST_FORWARD_DAYS
RETURN_COLUMN = "future_5d_high_close_to_low_open_return"
SUMMARY_COLUMNS = ["row_type", "month", "as_of_date", "strategy", "average_return"]


STRATEGIES = [
    ("botton", lianghua1_search_botton.run_search),
    ("test", lianghua1_search_test.run_search),
    ("turn", lianghua1_search_turn.run_search),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run monthly backtest for search strategies.")
    parser.add_argument(
        "-m",
        "--month",
        help="Backtest month in YYYYMM format. Defaults to config.get_end_date() month.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output CSV path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide strategy console output while running.",
    )
    return parser.parse_args()


def normalize_month(month=None):
    month = str(month or config.get_end_date()[:6]).strip()
    if not (month.isdigit() and len(month) == 6):
        raise ValueError(f"month must be YYYYMM, got: {month}")
    return month


def load_cached_trade_dates():
    if not DAILY_BASIC_DIR.exists():
        raise FileNotFoundError(f"daily_basic cache directory not found: {DAILY_BASIC_DIR}")

    dates = []
    for cache_file in DAILY_BASIC_DIR.glob("*.csv"):
        trade_date = cache_file.stem
        if trade_date.isdigit() and len(trade_date) == 8:
            dates.append(trade_date)

    if not dates:
        raise ValueError(f"no daily_basic cache dates found in {DAILY_BASIC_DIR}")

    return sorted(set(dates))


def build_backtest_dates(month, trade_dates, forward_days=BACKTEST_FORWARD_DAYS):
    month_dates = [trade_date for trade_date in trade_dates if trade_date.startswith(month)]
    if not month_dates:
        return []

    trade_date_set = set(trade_dates)
    backtest_dates = []
    for trade_date in month_dates:
        all_index = trade_dates.index(trade_date)
        future_dates = trade_dates[all_index + 1 : all_index + 1 + forward_days]
        if len(future_dates) < forward_days:
            continue

        if all(future_date in trade_date_set for future_date in future_dates):
            backtest_dates.append(trade_date)

    return backtest_dates


def run_strategy(strategy_name, run_search, trade_date, quiet=False):
    kwargs = {
        "as_of_date": trade_date,
        "pro": object(),
        "save_file": False,
    }

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            result_df = run_search(**kwargs)
    else:
        result_df = run_search(**kwargs)

    if result_df is None or result_df.empty or RETURN_COLUMN not in result_df.columns:
        return pd.DataFrame()

    result_df = result_df.copy()
    result_df[RETURN_COLUMN] = pd.to_numeric(result_df[RETURN_COLUMN], errors="coerce")
    result_df = result_df.dropna(subset=[RETURN_COLUMN])
    if result_df.empty:
        return pd.DataFrame()

    result_df.insert(0, "strategy", strategy_name)
    result_df.insert(1, "as_of_date", trade_date)
    result_df.insert(2, "month", trade_date[:6])
    result_df["future_5d"] = result_df[RETURN_COLUMN]
    return result_df


def build_daily_summary(detail_df, month, backtest_dates):
    strategy_names = [strategy_name for strategy_name, _ in STRATEGIES]
    daily_index = pd.MultiIndex.from_product(
        [[month], backtest_dates, strategy_names],
        names=["month", "as_of_date", "strategy"],
    )
    daily_summary = daily_index.to_frame(index=False)

    if not detail_df.empty:
        observed_summary = (
            detail_df.groupby(["month", "as_of_date", "strategy"], as_index=False)
            .agg(average_return=("future_5d", "mean"))
        )
        daily_summary = daily_summary.merge(
            observed_summary,
            on=["month", "as_of_date", "strategy"],
            how="left",
        )
    else:
        daily_summary["average_return"] = float("nan")

    daily_summary["average_return"] = pd.to_numeric(daily_summary["average_return"], errors="coerce")
    daily_summary.insert(0, "row_type", "daily_strategy_average")
    return daily_summary[SUMMARY_COLUMNS]


def build_monthly_summary(daily_summary, month):
    strategy_names = [strategy_name for strategy_name, _ in STRATEGIES]
    if daily_summary.empty:
        monthly_summary = pd.DataFrame({"strategy": strategy_names})
        monthly_summary["average_return"] = float("nan")
    else:
        monthly_summary = (
            daily_summary.groupby("strategy", as_index=False)["average_return"]
            .mean()
            .reindex(columns=["strategy", "average_return"])
        )
        monthly_summary = (
            pd.DataFrame({"strategy": strategy_names})
            .merge(monthly_summary, on="strategy", how="left")
        )

    monthly_summary.insert(0, "as_of_date", "")
    monthly_summary.insert(0, "month", month)
    monthly_summary.insert(0, "row_type", "monthly_strategy_average")
    return monthly_summary[SUMMARY_COLUMNS]


def build_output(daily_summary, monthly_summary):
    output_df = pd.concat([daily_summary, monthly_summary], ignore_index=True)
    output_df["average_return"] = pd.to_numeric(output_df["average_return"], errors="coerce").round(3)
    return output_df


def default_output_file(month):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"strategy_backtest_{month}_{timestamp}.csv"


def run_monthly_backtest(month=None, output_file=None, quiet=False):
    month = normalize_month(month)
    trade_dates = load_cached_trade_dates()
    backtest_dates = build_backtest_dates(month, trade_dates)
    if not backtest_dates:
        raise ValueError(f"{month} has no dates with at least {BACKTEST_FORWARD_DAYS} future trade days in cache")

    detail_frames = []
    total_tasks = len(backtest_dates) * len(STRATEGIES)
    task_index = 0

    for trade_date in backtest_dates:
        for strategy_name, run_search in STRATEGIES:
            task_index += 1
            print(f"[monthly-backtest] {task_index}/{total_tasks} {trade_date} {strategy_name}")
            strategy_df = run_strategy(strategy_name, run_search, trade_date, quiet=quiet)
            if not strategy_df.empty:
                detail_frames.append(strategy_df)

    if detail_frames:
        detail_df = pd.concat(detail_frames, ignore_index=True)
    else:
        detail_df = pd.DataFrame(columns=["strategy", "as_of_date", "month", "future_5d"])

    daily_summary = build_daily_summary(detail_df, month, backtest_dates)
    monthly_summary = build_monthly_summary(daily_summary, month)
    output_df = build_output(daily_summary, monthly_summary)

    output_file = Path(output_file) if output_file else default_output_file(month)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_file, index=False, encoding="utf-8-sig", float_format="%.3f")

    print(f"[monthly-backtest] month={month}")
    print(f"[monthly-backtest] backtest_dates={len(backtest_dates)} ({backtest_dates[0]}-{backtest_dates[-1]})")
    print(f"[monthly-backtest] detail_rows={len(detail_df)}")
    print(f"[monthly-backtest] saved: {output_file}")
    return output_df


def main():
    args = parse_args()
    run_monthly_backtest(month=args.month, output_file=args.output, quiet=args.quiet)


if __name__ == "__main__":
    main()
