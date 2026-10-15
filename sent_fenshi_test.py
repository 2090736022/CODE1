from __future__ import annotations

from collections import deque
from datetime import datetime, time as datetime_time
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr
import queue
import smtplib
import threading
import time
import tkinter as tk
from tkinter import ttk

from fenshi import get_minute_em


SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
SMTP_USER = "2972038373@qq.com"
SMTP_PASSWORD = "2972038373@qq.com"

SENDER_NAME = "price alert"
RECEIVER_EMAIL = "2972038373@qq.com"

STOCK_CODE = "000811"
PRICE_CHANGE_THRESHOLD = 0.01
CHECK_INTERVAL_SECONDS = 8
NON_TRADING_CHECK_INTERVAL_SECONDS = 10

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 120
MARGIN_LEFT = 16
MARGIN_BOTTOM = 72

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
    return any(start <= current_time <= end for start, end in TRADING_TIME_RANGES)


def get_latest_minute_close(stock_code: str) -> tuple[str, float, float | None]:
    trade_date = today_str()
    data, change_rate = get_minute_em(stock_code, trade_date)
    if data is None or data.empty:
        raise ValueError(f"{stock_code} has no minute data on {trade_date}")

    latest = data.tail(1).iloc[0]
    close_column = next((name for name in ("close", "收盘", "最新价") if name in data.columns), None)
    time_column = next((name for name in ("day", "时间", "日期时间", "datetime") if name in data.columns), None)

    if close_column is None:
        raise ValueError(f"missing close column, columns={list(data.columns)}")

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
        f"Stock: {stock_code}\n"
        f"Time: {minute_time}\n"
        f"Close: {close_price}\n"
        f"Change: {change_rate_text}"
    )


def send_email(subject: str, content: str) -> None:
    message = MIMEText(content, "plain", "utf-8")
    message["From"] = formataddr((Header(SENDER_NAME, "utf-8").encode(), SMTP_USER))
    message["To"] = RECEIVER_EMAIL
    message["Subject"] = Header(subject, "utf-8").encode()

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, [RECEIVER_EMAIL], message.as_string())


def should_send_email(current_price: float, base_price: float) -> bool:
    if base_price <= 0:
        return False
    return abs(current_price - base_price) / base_price >= PRICE_CHANGE_THRESHOLD


def place_bottom_left(root: tk.Tk) -> None:
    root.update_idletasks()
    screen_height = root.winfo_screenheight()
    x = MARGIN_LEFT
    y = max(0, screen_height - WINDOW_HEIGHT - MARGIN_BOTTOM)
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")


def build_overlay(root: tk.Tk) -> tuple[tk.StringVar, tk.StringVar]:
    root.title("minute alert")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    root.attributes("-toolwindow", True)

    container = ttk.Frame(root, padding=10)
    container.pack(fill="both", expand=True)

    ttk.Label(container, text="minute alert").pack(anchor="w")

    line1 = tk.StringVar(value="waiting...")
    line2 = tk.StringVar(value="")
    ttk.Label(container, textvariable=line1).pack(anchor="w", pady=(8, 2))
    ttk.Label(container, textvariable=line2).pack(anchor="w")

    controls = ttk.Frame(container)
    controls.pack(fill="x", side="bottom", pady=(8, 0))
    ttk.Button(controls, text="Close", command=root.destroy).pack(side="right")
    return line1, line2


def format_display_line(stock_code: str, minute_time: str, close_price: float, change_rate: float | None) -> str:
    change_text = "--" if change_rate is None else f"{change_rate:.2f}%"
    refresh_time = datetime.now().strftime("%H:%M:%S")
    return f"{stock_code} {minute_time} close={close_price:.3f} change={change_text} refresh={refresh_time}"


def polling_worker(out_queue: queue.Queue[str]) -> None:
    last_sent_price = None
    last_seen_time = None
    history: deque[str] = deque(maxlen=2)

    while True:
        if not is_trading_time():
            time.sleep(NON_TRADING_CHECK_INTERVAL_SECONDS)
            continue

        try:
            minute_time, close_price, change_rate = get_latest_minute_close(STOCK_CODE)
        except Exception as exc:
            out_queue.put(f"fetch failed: {exc}")
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        is_new_minute = minute_time != last_seen_time
        last_seen_time = minute_time
        line = format_display_line(STOCK_CODE, minute_time, close_price, change_rate)
        if is_new_minute:
            history.append(line)
        elif history:
            history[-1] = line
        else:
            history.append(line)
        out_queue.put("\n".join(history))
        print(line)

        if last_sent_price is None:
            last_sent_price = close_price
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        if should_send_email(close_price, last_sent_price):
            subject = f"{STOCK_CODE} wave alert"
            content = build_email_content(STOCK_CODE, minute_time, close_price, change_rate)
            try:
                send_email(subject, content)
                last_sent_price = close_price
                out_queue.put("email sent")
                print("email sent")
            except Exception as exc:
                #out_queue.put(f"email failed: {exc}")
                #print(f"email failed: {exc}")
                pass

        time.sleep(CHECK_INTERVAL_SECONDS)


def main() -> None:
    root = tk.Tk()
    line1_var, line2_var = build_overlay(root)
    place_bottom_left(root)

    msg_queue: queue.Queue[str] = queue.Queue()
    threading.Thread(target=polling_worker, args=(msg_queue,), daemon=True).start()

    def pump_queue() -> None:
        latest: list[str] = []
        try:
            while True:
                latest.append(msg_queue.get_nowait())
        except queue.Empty:
            pass

        if latest:
            parts = latest[-1].splitlines()
            if len(parts) >= 2:
                line1_var.set(parts[-2])
                line2_var.set(parts[-1])
            else:
                line1_var.set(parts[-1])
                line2_var.set("")

        if root.winfo_exists():
            root.after(500, pump_queue)

    pump_queue()
    root.mainloop()


if __name__ == "__main__":
    main()
