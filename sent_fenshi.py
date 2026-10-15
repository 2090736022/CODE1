
from datetime import datetime, time as datetime_time
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr
import smtplib
import time

from fenshi import get_minute_em


# QQ SMTP config.
# SMTP_PASSWORD is the QQ Mail SMTP authorization code, not your QQ password.
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
SMTP_USER = "2972038373@qq.com"
SMTP_PASSWORD = "ggzzgbrylvuddfjg"

SENDER_NAME = "股价提醒"
RECEIVER_EMAIL = "2972038373@qq.com"

STOCK_CODE = "600255"
PRICE_CHANGE_THRESHOLD = 0.01
CHECK_INTERVAL_SECONDS = 10
NON_TRADING_CHECK_INTERVAL_SECONDS = 60

TRADING_TIME_RANGES = (
    (datetime_time(9, 25), datetime_time(11, 30)),
    (datetime_time(13, 0), datetime_time(15, 0)),
)


def today_str() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def is_trading_time(now: datetime | None = None) -> bool:
    current = now or datetime.now()
    if current.weekday() >= 5:
        return False

    current_time = current.time()
    return any(
        start <= current_time <= end
        for start, end in TRADING_TIME_RANGES
    )


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
    return change_rate >= PRICE_CHANGE_THRESHOLD


def main() -> None:
    last_sent_price = None
    last_seen_time = None

    while True:
        if not is_trading_time():
            time.sleep(NON_TRADING_CHECK_INTERVAL_SECONDS)
            continue

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