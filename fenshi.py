from datetime import datetime

import akshare as ak
import tushare as ts

import config


def format_em_symbol(symbol):
    symbol = str(symbol).strip().lower()
    if symbol.startswith(("sz", "sh")):
        return symbol
    if symbol.startswith("00"):
        return f"sz{symbol}"
    if symbol.startswith("60"):
        return f"sh{symbol}"
    raise ValueError(f"不支持的股票代码: {symbol}")


def format_ts_code(symbol):
    symbol = str(symbol).strip().lower()
    if symbol.startswith("sz"):
        return f"{symbol[2:]}.SZ"
    if symbol.startswith("sh"):
        return f"{symbol[2:]}.SH"
    if symbol.startswith("00"):
        return f"{symbol}.SZ"
    if symbol.startswith("60"):
        return f"{symbol}.SH"
    raise ValueError(f"不支持的股票代码: {symbol}")


def format_trade_date(date):
    if date:
        return str(date).replace("-", "")
    return datetime.today().strftime("%Y%m%d")


def get_close_column(df):
    close_column = next(
        (name for name in ("close", "收盘", "最新价") if name in df.columns),
        None,
    )
    if close_column is None:
        raise ValueError(f"分时数据里没有 close/收盘/最新价 字段，实际字段: {list(df.columns)}")
    return close_column


def get_previous_trade_close(symbol, trade_date=None):
    ts.set_token(config.TUSHARE_TOKEN)
    pro = ts.pro_api()
    ts_code = format_ts_code(symbol)
    end_date = format_trade_date(trade_date)

    df = pro.daily(ts_code=ts_code, end_date=end_date, limit=2)
    if df is None or df.empty:
        raise ValueError(f"{ts_code} 没有拉取到上一交易日收盘价")

    df = df.sort_values("trade_date", ascending=False)
    if len(df) >= 2 and str(df.iloc[0]["trade_date"]) == end_date:
        return float(df.iloc[1]["close"])

    return float(df.iloc[0]["close"])


def get_change_rate(latest_close, previous_close):
    if previous_close <= 0:
        return None
    return (latest_close / previous_close - 1) * 100


def get_minute_em(symbol, date=None):
    """东方财富当日 1 分钟分时数据，并按上一交易日收盘价计算涨跌幅。"""
    df = ak.stock_zh_a_minute(
        symbol=format_em_symbol(symbol),
        period="1",
        adjust="",
    )
    #print(df)
    if df is None or df.empty:
        return df, None

    close_column = get_close_column(df)
    latest_close = float(df.tail(1).iloc[0][close_column])
    previous_close = get_previous_trade_close(symbol, date)
    change_rate = get_change_rate(latest_close, previous_close)

    return df, change_rate

