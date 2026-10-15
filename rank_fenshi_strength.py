# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import time
from pathlib import Path

import akshare as ak
import pandas as pd

from fenshi_strength import calculate_strength_for_stock


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STOCK_FILE = BASE_DIR / "number.csv"
DEFAULT_DAILY_CACHE_DIR = BASE_DIR / "model_artifacts" / "download_daily" / "daily"
DEFAULT_OUTPUT_FILE = BASE_DIR / "fenshi_strength_top10.csv"


def cache_code(ts_code: str) -> str:
    return str(ts_code).strip().replace(".", "_")


def em_symbol_from_ts_code(ts_code: str) -> str:
    code, market = str(ts_code).strip().split(".")
    market = market.upper()
    if market == "SZ":
        return f"sz{code}"
    if market == "SH":
        return f"sh{code}"
    raise ValueError(f"unsupported ts_code: {ts_code}")


def normalize_trade_date(value: str | None) -> str | None:
    if not value:
        return None
    return str(value).strip().replace("-", "")


def read_stock_list(stock_file: Path) -> pd.DataFrame:
    stocks = pd.read_csv(stock_file, dtype={"ts_code": str, "symbol": str})
    if "ts_code" not in stocks.columns:
        raise ValueError("number.csv 缺少 ts_code 列")
    if "name" not in stocks.columns:
        stocks["name"] = ""
    return stocks.dropna(subset=["ts_code"]).copy()


def read_daily_row(ts_code: str, daily_cache_dir: Path, trade_date: str | None) -> pd.Series | None:
    cache_file = daily_cache_dir / f"{cache_code(ts_code)}.csv"
    if not cache_file.exists():
        return None

    daily = pd.read_csv(cache_file, dtype={"ts_code": str, "trade_date": str})
    if daily.empty or "trade_date" not in daily.columns or "pct_chg" not in daily.columns:
        return None

    daily = daily.sort_values("trade_date").copy()
    if trade_date:
        daily = daily[daily["trade_date"] == trade_date]
    if daily.empty:
        return None

    return daily.iloc[-1]


def get_minute_data(em_symbol: str) -> pd.DataFrame:
    return ak.stock_zh_a_minute(
        symbol=em_symbol,
        period="1",
        adjust="",
    )


def result_to_row(result, ts_code: str, name: str, pct_chg: float) -> dict[str, object]:
    return {
        "rank": None,
        "ts_code": ts_code,
        "name": name,
        "trade_date": result.trade_date,
        "pct_chg": pct_chg,
        "final_score": result.final_score,
        "rule_score": result.rule_score,
        "buy_strength": result.buy_strength,
        "big_volume_buy_strength": result.big_volume_buy_strength,
        "recent_buy_strength": result.recent_buy_strength,
        "volume_pile_ratio": result.volume_pile_ratio,
        "volume_pile_bias": result.volume_pile_bias,
        "volume_pile_reaction": result.volume_pile_reaction,
        "post_pile_pressure": result.post_pile_pressure,
        "latest_close": result.latest_close,
    }


def rank_fenshi_strength(
    stock_file: Path,
    daily_cache_dir: Path,
    output_file: Path,
    trade_date: str | None,
    min_pct_chg: float,
    top_n: int,
    request_sleep: float,
) -> pd.DataFrame:
    stocks = read_stock_list(stock_file)
    rows: list[dict[str, object]] = []
    skipped_low = 0
    skipped_missing_daily = 0
    failed = 0

    total = len(stocks)
    for index, stock in stocks.iterrows():
        ts_code = str(stock["ts_code"]).strip()
        name = str(stock.get("name", "")).strip()
        daily_row = read_daily_row(ts_code, daily_cache_dir, trade_date)
        if daily_row is None:
            skipped_missing_daily += 1
            continue

        pct_chg = pd.to_numeric(daily_row.get("pct_chg"), errors="coerce")
        if pd.isna(pct_chg) or float(pct_chg) < min_pct_chg:
            skipped_low += 1
            continue

        target_date = str(daily_row["trade_date"])
        try:
            em_symbol = em_symbol_from_ts_code(ts_code)
            minute_data = get_minute_data(em_symbol)
            result = calculate_strength_for_stock(
                stock_code=em_symbol,
                trade_date=target_date,
                minute_data=minute_data,
            )
            rows.append(result_to_row(result, ts_code, name, float(pct_chg)))
        except Exception as error:
            failed += 1
            print(f"{ts_code} {name} 计算失败: {error}")

        if request_sleep > 0:
            time.sleep(request_sleep)

        if len(rows) % 10 == 0 and rows:
            print(f"已计算 {len(rows)} 只，当前扫描进度 {index + 1}/{total}")

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(
            ["final_score", "volume_pile_ratio", "pct_chg"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        result_df["rank"] = result_df.index + 1

    top_df = result_df.head(top_n)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"本地日K缺失跳过: {skipped_missing_daily}")
    print(f"涨幅小于 {min_pct_chg:.2f}% 跳过: {skipped_low}")
    print(f"分时计算失败: {failed}")
    print(f"结果已保存: {output_file}")
    return top_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量计算分时成交强度并输出前十名")
    parser.add_argument("--stock-file", type=Path, default=DEFAULT_STOCK_FILE)
    parser.add_argument("--daily-cache-dir", type=Path, default=DEFAULT_DAILY_CACHE_DIR)
    parser.add_argument("-d", "--date", help="交易日期，格式 YYYYMMDD 或 YYYY-MM-DD；不填则用每只股票缓存里的最新日K")
    parser.add_argument("--min-pct-chg", type=float, default=7.0, help="日K涨幅过滤阈值，默认 2%%")
    parser.add_argument("--top", type=int, default=10, help="输出排名数量，默认 10")
    parser.add_argument("--sleep", type=float, default=0.05, help="分时请求间隔秒数，默认 0.05")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_df = rank_fenshi_strength(
        stock_file=args.stock_file,
        daily_cache_dir=args.daily_cache_dir,
        output_file=args.output,
        trade_date=normalize_trade_date(args.date),
        min_pct_chg=args.min_pct_chg,
        top_n=args.top,
        request_sleep=args.sleep,
    )

    if top_df.empty:
        print("没有符合条件的股票。")
        return

    columns = ["rank", "ts_code", "name", "trade_date", "pct_chg", "final_score", "volume_pile_ratio", "volume_pile_bias", "volume_pile_reaction"]
    print(top_df[columns].to_string(index=False))


if __name__ == "__main__":
    main()
