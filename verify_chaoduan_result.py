"""重新拉取腾讯分时，核对已保存结果的达标时刻和成交量。"""
import pandas as pd
import config
from minute_data import fetch_cumulative_minutes


def main():
    path = config.RESULT_DIR / f"超短筛选结果_{config.get_end_date()}.csv"
    rows = pd.read_csv(path, dtype={"ts_code": str, "trade_date": str, "minute_trade_date": str})
    for _, row in rows.iterrows():
        data = fetch_cumulative_minutes(row.ts_code).set_index("datetime")["cumulative_volume"]
        current = data.loc[pd.Timestamp(row.minute_trade_date + " " + row.minute_match_time)]
        previous = data.loc[pd.Timestamp(row.trade_date + " " + row.minute_match_time)]
        same = current == row.cumulative_volume and previous == row.previous_cumulative_volume
        print(row.ts_code, row.minute_match_time, row.minute_trade_date, current,
              row.trade_date, previous, "ratio=", round(current / previous, 6),
              "saved_match=", same, flush=True)
        if not same:
            raise ValueError(f"{row.ts_code} 最新接口数据与已保存结果不一致")


if __name__ == "__main__":
    main()
