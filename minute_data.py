"""免费新浪一分钟 K 线；成交量统一为手，完整数据按日缓存。"""
import json
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent / "model_artifacts" / "minute_sina"
CUMULATIVE_CACHE_DIR = Path(__file__).resolve().parent / "model_artifacts" / "minute_tencent_cumulative"


def fetch_cumulative_minutes(ts_code):
    """腾讯近五日分时的当日累计量（手），09:30 已含开盘集合竞价。

    09:15 起的委托量不是成交量；直接使用供应商累计字段，不再求和。
    """
    code, market = ts_code.upper().split(".")
    if market not in {"SH", "SZ"}:
        raise ValueError(f"不支持的市场: {ts_code}")
    symbol = market.lower() + code
    response = requests.get(
        "https://web.ifzq.gtimg.cn/appstock/app/day/query",
        params={"code": symbol}, headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise ValueError(f"{ts_code} 腾讯分时返回错误")
    rows = []
    for day in payload.get("data", {}).get(symbol, {}).get("data", []):
        for line in day["data"]:
            parts = line.split()
            rows.append((pd.to_datetime(day["date"] + parts[0], format="%Y%m%d%H%M"), float(parts[2]), float(parts[1])))
    data = pd.DataFrame(rows, columns=["datetime", "cumulative_volume", "price"])
    if data.empty:
        raise ValueError(f"{ts_code} 腾讯累计成交量为空")
    return data.sort_values("datetime").reset_index(drop=True)


def cumulative_window(data, trade_date, start, end):
    result = window(data, trade_date, start, end).sort_values("datetime")
    volume = pd.to_numeric(result["cumulative_volume"], errors="raise")
    if volume.isna().any() or not volume.map(lambda v: 0 <= v < float("inf")).all() or (volume.diff().dropna() < 0).any():
        raise ValueError("当日累计成交量必须有效、非负且不递减")
    result["cumulative_volume"] = volume
    price = pd.to_numeric(result["price"], errors="raise")
    if not price.map(lambda v: 0 < v < float("inf")).all():
        raise ValueError("分钟价格无效")
    result["price"] = price
    return result


def get_cumulative_window(ts_code, trade_date, start="09:45:00", end="10:00:00"):
    date = pd.to_datetime(str(trade_date)).strftime("%Y%m%d")
    folder = CUMULATIVE_CACHE_DIR / ts_code.replace(".", "_")
    cache = folder / f"{date}.csv"
    if cache.exists():
        try:
            return cumulative_window(pd.read_csv(cache), date, start, end)
        except (ValueError, KeyError):
            pass
    data = fetch_cumulative_minutes(ts_code)
    folder.mkdir(parents=True, exist_ok=True)
    for day, frame in data.groupby(data["datetime"].dt.strftime("%Y%m%d")):
        try:
            cumulative_window(frame, day, start, end)
        except ValueError:
            continue
        frame.to_csv(folder / f"{day}.csv", index=False, encoding="utf-8-sig")
    return cumulative_window(data, date, start, end)


def fetch_recent_minutes(ts_code):
    code, market = ts_code.upper().split(".")
    if market not in {"SH", "SZ"}:
        raise ValueError(f"不支持的市场: {ts_code}")
    response = requests.get(
        "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_data=/CN_MarketDataService.getKLineData",
        params={"symbol": market.lower() + code, "scale": "1", "ma": "no", "datalen": "1023"},
        headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
    )
    response.raise_for_status()
    body = response.text
    begin, end = body.find("(["), body.rfind("])")
    if begin < 0 or end < begin:
        raise ValueError(f"{ts_code} 新浪没有返回分钟 K 线: {body[:100]}")
    data = pd.DataFrame(json.loads(body[begin + 1:end + 1]))
    if data.empty:
        raise ValueError(f"{ts_code} 新浪分钟数据为空")
    data = data.rename(columns={"day": "datetime"})
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[["open", "high", "low", "close", "volume", "amount"]].isna().any().any() or (data["volume"] < 0).any():
        raise ValueError(f"{ts_code} 分钟行情含无效数值")
    data["volume"] /= 100  # 新浪为股，策略统一用手。
    return data.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


def window(data, trade_date, start, end):
    date = pd.to_datetime(str(trade_date)).strftime("%Y-%m-%d")
    times = pd.to_datetime(data["datetime"])
    expected = pd.date_range(f"{date} {start}", f"{date} {end}", freq="min")
    result = data.loc[times.isin(expected)].copy()
    if len(expected) == 0 or len(result) != len(expected) or not expected.isin(pd.to_datetime(result["datetime"])).all():
        raise ValueError(f"{trade_date} {start}-{end} 分钟数据不完整；近期接口不支持任意历史日期")
    return result


def full_day(data, trade_date):
    """校验全天 238 根，收盘集合竞价仅使用 15:00。"""
    return pd.concat([
        window(data, trade_date, "09:31:00", "11:30:00"),
        window(data, trade_date, "13:01:00", "14:57:00"),
        window(data, trade_date, "15:00:00", "15:00:00"),
    ], ignore_index=True).sort_values("datetime").reset_index(drop=True)


def get_full_day(ts_code, trade_date, refresh_attempts=2):
    """缺线时重新提取整日数据；只有通过完整性检查才覆盖缓存。"""
    date = pd.to_datetime(str(trade_date)).strftime("%Y%m%d")
    cache = CACHE_DIR / ts_code.replace(".", "_") / f"{date}.csv"
    if cache.exists():
        try:
            return full_day(pd.read_csv(cache), date)
        except (ValueError, KeyError):
            print(f"{ts_code} {date} 全天分钟缓存不完整，重新提取")
    error = None
    for attempt in range(refresh_attempts):
        try:
            result = full_day(fetch_recent_minutes(ts_code), date)
        except (ValueError, KeyError, requests.RequestException) as exc:
            error = exc
            print(f"{ts_code} {date} 重新提取第 {attempt + 1} 次失败: {exc}")
            continue
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache, index=False, encoding="utf-8-sig")
        return result
    raise ValueError(f"{ts_code} {date} 重新提取 {refresh_attempts} 次后仍无法取得完整的 238 根分钟 K 线: {error}") from error


def get_window(ts_code, trade_date, start="09:45:00", end="10:00:00"):
    date = pd.to_datetime(str(trade_date)).strftime("%Y%m%d")
    folder = CACHE_DIR / ts_code.replace(".", "_")
    cache = folder / f"{date}.csv"
    if cache.exists():
        try:
            return window(pd.read_csv(cache), date, start, end)
        except (ValueError, KeyError):
            pass
    data = fetch_recent_minutes(ts_code)
    folder.mkdir(parents=True, exist_ok=True)
    for day, frame in data.groupby(data["datetime"].dt.strftime("%Y%m%d")):
        # 只缓存包含完整策略窗口的数据，不把未完成早盘永久保存。
        try:
            window(frame, day, start, end)
        except ValueError:
            continue
        frame.to_csv(folder / f"{day}.csv", index=False, encoding="utf-8-sig")
    return window(data, date, start, end)


if __name__ == "__main__":
    import config
    from chaoduan import minute_volume_matches
    for symbol in ("000001.SZ", "600000.SH", "600519.SH"):
        data = fetch_cumulative_minutes(symbol)
        selection = config.get_end_date()
        dates = sorted(data["datetime"].dt.strftime("%Y%m%d").unique())
        following = next(day for day in dates if day > selection)
        for day in (selection, following):
            frame = cumulative_window(data, day, "09:45:00", "10:00:00")
            actual = get_cumulative_window(symbol, day)
            assert actual["cumulative_volume"].tolist() == frame["cumulative_volume"].tolist()
            print(symbol, day, "bars=", len(frame), "cumulative at 09:46=", frame.iloc[1]["cumulative_volume"], flush=True)
        print("strategy/cache", symbol, minute_volume_matches(symbol, following, selection), flush=True)
