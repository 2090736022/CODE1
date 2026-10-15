"""主板非 ST：流通市值 > 20 亿、实际换手率 > 1.25%、均线条件及次日分钟信号。"""
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from minute_data import get_window, get_cumulative_window

from chaoduan import run_search as run_base_search
from chaoduan import minute_volume_matches, is_limit_up


def basic_matches(row, ts_code):
    # circ_mv 单位万元：20 亿元 = 200000 万元。
    circ_mv = pd.to_numeric(row.get("circ_mv"), errors="coerce")
    turnover = pd.to_numeric(row.get("turnover_rate_f"), errors="coerce")
    # 实际换手率缺失时不以普通换手率代替。
    return bool(pd.notna(circ_mv) and pd.notna(turnover)
                and 200_000 < circ_mv < float("inf")
                and 1.25 < turnover < float("inf"))


def is_one_price_limit_down(row):
    """主板非 ST：一字价等于昨收的 90%，跌停价四舍五入到分。"""
    prices = pd.to_numeric(pd.Series({key: row.get(key) for key in ("open", "high", "low", "close")}), errors="coerce")
    if not prices.map(lambda value: 0 < value < float("inf")).all():
        return False
    if not prices.eq(prices.iloc[0]).all():
        return False
    previous_close = pd.to_numeric(row.get("pre_close"), errors="coerce")
    if pd.notna(previous_close) and 0 < previous_close < float("inf"):
        limit_price = (Decimal(str(previous_close)) * Decimal("0.90")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return Decimal(str(prices.iloc[0])) == limit_price
    # 兼容仅有涨跌幅的日线数据源。
    pct_chg = pd.to_numeric(row.get("pct_chg"), errors="coerce")
    return bool(pd.notna(pct_chg) and -float("inf") < pct_chg <= -9.51)


def match_strategy(df):
    """选股日不能涨停或一字跌停，且 MA5 > MA10 或 MA10 > MA20。"""
    if df is None or len(df) < 20:
        return False
    data = df.sort_values("trade_date")
    if is_limit_up(data.iloc[-1]) or is_one_price_limit_down(data.iloc[-1]):
        return False
    close = pd.to_numeric(data["close"].tail(20), errors="coerce")
    if not close.map(lambda value: 0 < value < float("inf")).all():
        return False
    return bool(close.tail(5).mean() > close.tail(10).mean()
                or close.tail(10).mean() > close.mean())


def early_limit_up_filter(ts_code, observation_date, previous_close, *, observation_close, daily_volume_ratio):
    """09:31–09:39 触板后，收盘未封板或全日量比 > 2.1 时放行。"""
    previous_close = Decimal(str(previous_close))
    if not previous_close.is_finite() or previous_close <= 0:
        raise ValueError("昨收无效，无法检查早盘涨停")
    limit_price = float((previous_close * Decimal("1.10")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    bars = get_window(ts_code, observation_date, "09:31:00", "09:39:00")
    high = pd.to_numeric(bars["high"], errors="raise")
    if not high.map(lambda value: 0 < value < float("inf")).all():
        raise ValueError("分钟最高价无效，无法检查早盘涨停")
    hit = bars.loc[high >= limit_price].sort_values("datetime")
    if not hit.empty:
        close = float(observation_close)
        ratio = float(daily_volume_ratio)
        if not 0 < close < float("inf") or not 0 <= ratio < float("inf"):
            raise ValueError("判定日收盘价或全日量比无效")
        return close < limit_price or ratio > 2.1
    return True


def run_search(as_of_date=None, result_dir=None, stock_file=None, pro=None,
               request_sleep=0, save_file=True):
    """复用 chaoduan 的分钟信号、日期逻辑和成交量输出，单独保存结果。"""
    return run_base_search(
        as_of_date=as_of_date, result_dir=result_dir, stock_file=stock_file,
        pro=pro, request_sleep=request_sleep, save_file=save_file,
        daily_matcher=match_strategy, basic_matcher=basic_matches,
        minute_prefilter=early_limit_up_filter,
        observation_min_turnover=9.5,
        observation_positive_close=True,
        observation_min_volume_ratio=1.1,
        result_prefix="超短1筛选结果",
    )


if __name__ == "__main__":
    run_search()
