# -*- coding: utf-8 -*-

from datetime import datetime
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr
import smtplib
import time

#from fenshi import get_minute_em

from datetime import datetime

import akshare as ak
import tushare as ts

import config

ts.set_token('a14f37a303d0e280c839224d331315323a8cde36ffbe5c41757fef31')

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
    ts_code = "000001.SH"
    end_date = format_trade_date(trade_date)

    df = ts.pro_bar(ts_code=ts_code, asset='I',end_date=end_date, limit=2)
    #print(df)
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
        symbol="sh000001",
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



# QQ SMTP config.
# SMTP_PASSWORD is the QQ Mail SMTP authorization code, not your QQ password.
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
SMTP_USER = "2972038373@qq.com"
SMTP_PASSWORD = "ggzzgbrylvuddfjg"

SENDER_NAME = "股价提醒"
RECEIVER_EMAIL = "2972038373@qq.com"

STOCK_CODE = "000001"
PRICE_CHANGE_THRESHOLD = 0.01
CHECK_INTERVAL_SECONDS = 10


def today_str() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def get_latest_minute_close(stock_code: str) -> tuple[str, float, float | None]:
    trade_date = today_str()
    data, change_rate = get_minute_em(stock_code, trade_date)
    #print(stock_code)
    if data is None or data.empty:
        raise ValueError(f"{stock_code} 在 {trade_date} 没有拉取到分时数据")

    latest = data.tail(1).iloc[0]
    close_column = next(
        (name for name in ("close", "收盘", "最新价") if name in data.columns),
        None,
    )
    time_column = next(
        (name for name in ("day", "时间", "日期时间", "datetime") if name in data.columns),
        None,
    )

    if close_column is None:
        raise ValueError(f"分时数据里没有 close/收盘/最新价 字段，实际字段: {list(data.columns)}")

    close_price = float(latest[close_column])
    minute_time = str(latest[time_column]) if time_column else str(latest.name)

    return minute_time, close_price, change_rate


def build_email_content(
    stock_code: str,
    minute_time: str,
    close_price: float,
    change_rate: float | None,
) -> str:
    change_rate_text = "--" if change_rate is None else f"{change_rate:.2f}%"
    return (
        f"股票代码：{stock_code}\n"
        f"时间：{minute_time}\n"
        f"股价：{close_price}\n"
        f"涨跌幅：{change_rate_text}"
    )


def send_email(subject: str, content: str) -> None:
    message = MIMEText(content, "plain", "utf-8")
    message["From"] = formataddr(
        (Header(SENDER_NAME, "utf-8").encode(), SMTP_USER)
    )
    message["To"] = RECEIVER_EMAIL
    message["Subject"] = Header(subject, "utf-8").encode()

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, [RECEIVER_EMAIL], message.as_string())


def should_send_email(current_price: float, base_price: float) -> bool:
    if base_price <= 0:
        return False

    change_rate = abs(current_price - base_price) / base_price
    return change_rate >= 0.001


def main() -> None:
    last_sent_price = None
    last_seen_time = None

    while True:
        try:
            minute_time, close_price, change_rate = get_latest_minute_close(STOCK_CODE)
        except Exception as exc:
            print(f"分时数据拉取失败: {exc}")
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        print(f"{STOCK_CODE} {minute_time} close={close_price} change={change_rate}")

        if last_sent_price is None:
            last_sent_price = close_price
            last_seen_time = minute_time
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        if minute_time == last_seen_time:
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        last_seen_time = minute_time

        if should_send_email(close_price, last_sent_price):
            subject = f"{STOCK_CODE} 波动提醒"
            content = build_email_content(STOCK_CODE, minute_time, close_price, change_rate)
            send_email(subject, content)
            last_sent_price = close_price
            print("邮件发送成功")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
