import re
import time
from datetime import datetime, timedelta
from pathlib import Path
import akshare as ak
import pandas as pd
# import tushare as ts

import config
# from lianghua1_search import TOKEN, create_pro
from fenshi import get_latest_price_and_previous_volume
from progress import show_progress


SEARCH_RESULT_PATTERN = re.compile(r'量化筛选结果(\d{8})\.csv$')
MAX_MONTH_GAIN_BEFORE_BUY = config.MAX_MONTH_GAIN_BEFORE_BUY
MONTH_LOOKBACK_DAYS = config.MONTH_LOOKBACK_DAYS

def get_minute_em(symbol, date):
    """东方财富 当日1分钟分时（稳定）"""
    df = ak.stock_zh_a_hist_min_em(
        symbol=symbol,
        period="1",             # 1分钟
        start_date=f"{date} 09:30:00",
        end_date=f"{date} 15:00:00"
    )
    return df

def get_start_date(search_date):
    """向筛选日前多取一段日线，确保能判断买点和买入前一个月涨幅。"""
    date_value = datetime.strptime(search_date, '%Y%m%d')
    return (date_value - timedelta(days=config.BUY_LOOKBACK_DAYS_BEFORE_SEARCH)).strftime('%Y%m%d')


def load_csv_or_empty(file_path, columns):
    """读取 CSV；文件不存在或为空时返回指定列的空 DataFrame。"""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)

    return pd.read_csv(file_path)


def load_search_results(result_dir, only_date=None):
    """读取 result.csv 股票池，返回待判断买点的股票。"""
    search_results = []
    seen = set()
    result_dir = Path(result_dir)
    pool_file = result_dir / 'result.csv'

    if not pool_file.exists():
        print(f'股票池文件不存在: {pool_file}')
        return search_results

    df = pd.read_csv(pool_file)

    if 'ts_code' not in df.columns:
        print(f'跳过 {pool_file.name}: 没有 ts_code 列')
        return search_results

    for ts_code in df['ts_code'].dropna():
        ts_code = str(ts_code).strip()

        if not ts_code or ts_code in seen:
            continue

        seen.add(ts_code)
        search_results.append({
            'ts_code': ts_code,
        })

    return search_results


def month_gain_before_buy_over_100(df, buy_index):
    """
    最后买入过滤：
    截至买入前一天，如果近一个月涨幅超过 100%，取消买入。
    这里“近一个月”按 22 个交易日前的收盘价近似。
    """
    prev_index = buy_index - 1
    month_start_index = prev_index - config.MONTH_LOOKBACK_DAYS

    if month_start_index < 0:
        return False

    prev_close = df.iloc[prev_index]['close']
    month_start_close = df.iloc[month_start_index]['close']

    if month_start_close <= 0:
        return False

    month_gain = prev_close / month_start_close - 1
    return month_gain > config.MAX_MONTH_GAIN_BEFORE_BUY


def find_buy_with_tushare(pro, ts_code, search_date, end_date):
    """用 Tushare 日线数据回看买点，供 delete 判断历史买点使用。"""
    print(search_date)
    ts_code='603618.SH'
    df = pro.daily(
        ts_code=ts_code,
        start_date=get_start_date(search_date),
        end_date=end_date
    )

    if df is None or len(df) < 2:
        return None

    df = df.sort_values('trade_date').reset_index(drop=True)
    
    for index in range(1, len(df)):
        current_day = df.iloc[index]
        prev_day = df.iloc[index - 1]

        if current_day['trade_date'] <= search_date:
            continue

        is_yin = current_day['close'] < prev_day['close']
        is_shrink_volume = current_day['vol'] < prev_day['vol']

        if not (is_yin and is_shrink_volume):
            continue

        if month_gain_before_buy_over_100(df, index):
            print(f'{ts_code} {current_day["trade_date"]} 买入前一个月涨幅超过100%，取消买入')
            return None

        return {
            'record_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'buy_date': current_day['trade_date'],
            'ts_code': ts_code,
            'buy_price': current_day['close'],
        }

    return None


def find_buy_with_fenshi(ts_code, end_date):
    """用 fenshi.py 的当天分时数据判断买点，供 buy 当前运行使用。"""
    fenshi_df = get_latest_price_and_previous_volume(ts_code)

    if fenshi_df is None or fenshi_df.empty:
        return None

    current_data = fenshi_df.iloc[-1]

    if pd.isna(current_data['price']):
        return None

    return {
        'record_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'buy_date': end_date,
        'ts_code': current_data['ts_code'],
        'buy_price': current_data['price'],
    }


def find_first_shrink_yin_after_search(pro, ts_code, search_date, end_date):
    """
    买点规则：
    当前版本不再使用 Tushare 日线数据。
    买入数据来自 fenshi.get_latest_price_and_previous_volume，只使用当天分时数据。
    """
    # 原 Tushare 日线取数逻辑保留，不再使用：
    # df = pro.daily(
    #     ts_code=ts_code,
    #     start_date=get_start_date(search_date or end_date),
    #     end_date=end_date
    # )
    #
    # if df is None or len(df) < 2:
    #     return None
    #
    # df = df.sort_values('trade_date').reset_index(drop=True)
    #
    # for index in range(1, len(df)):
    #     current_day = df.iloc[index]
    #     prev_day = df.iloc[index - 1]
    #
    #     if search_date and current_day['trade_date'] <= search_date:
    #         continue
    #
    #     is_yin = current_day['close'] < prev_day['close']
    #     is_shrink_volume = current_day['vol'] < prev_day['vol']
    #
    #     if not (is_yin and is_shrink_volume):
    #         continue
    #
    #     if month_gain_before_buy_over_100(df, index):
    #         print(f'{ts_code} {current_day["trade_date"]} 买入前一个月涨幅超过100%，取消买入')
    #         return None
    #
    #     return {
    #         'record_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    #         'buy_date': current_day['trade_date'],
    #         'ts_code': ts_code,
    #         'buy_price': current_day['close'],
    #     }

    return find_buy_with_fenshi(ts_code, end_date)


def run_buy(
    end_date=None,
    result_dir=None,
    only_search_date=None,
    pro=None,
    request_sleep=0.0,
):
    """执行买点判断，并更新 records.csv 和 持仓.csv。"""
    # 原 Tushare 初始化逻辑保留，不再使用：
    # ts.set_token(config.TUSHARE_TOKEN)
    # pro = pro or create_pro()

    end_date = end_date or config.get_end_date()
    result_dir = Path(result_dir or config.RESULT_DIR)
    records_file = result_dir / 'records.csv'
    positions_file = result_dir / '持仓.csv'

    new_records = []

    search_results = load_search_results(result_dir, only_date=only_search_date)
    total_items = len(search_results)
    show_progress(0, total_items, 'buy 判断买点')

    for index, item in enumerate(search_results, start=1):
        ts_code = item['ts_code']
        search_date = item.get('search_date')

        try:
            buy_record = find_first_shrink_yin_after_search(pro, ts_code, search_date, end_date)

            if buy_record is None:
                print(f'{ts_code} 暂无买点')
                continue

            new_records.append(buy_record)
            print(f'{ts_code} 买入日期: {buy_record["buy_date"]}, 买入价格: {buy_record["buy_price"]}')

            if request_sleep:
                time.sleep(request_sleep)
        except Exception as e:
            print(f'{ts_code} 买点判断失败: {e}')
        finally:
            show_progress(index, total_items, 'buy 判断买点')

    result_dir.mkdir(exist_ok=True)

    record_columns = ['record_time', 'buy_date', 'ts_code', 'buy_price']
    records_df = load_csv_or_empty(records_file, record_columns)
    new_records_df = pd.DataFrame(new_records, columns=record_columns)

    all_records_df = pd.concat([records_df, new_records_df], ignore_index=True)
    all_records_df = all_records_df.drop_duplicates(subset=['buy_date', 'ts_code'], keep='first')
    all_records_df.to_csv(records_file, index=False, encoding='utf-8-sig')

    position_columns = ['ts_code', 'buy_date', 'buy_price']
    positions_df = load_csv_or_empty(positions_file, position_columns)
    new_positions_df = new_records_df[position_columns].copy()

    all_positions_df = pd.concat([positions_df, new_positions_df], ignore_index=True)
    all_positions_df = all_positions_df.drop_duplicates(subset=['ts_code'], keep='first')
    all_positions_df.to_csv(positions_file, index=False, encoding='utf-8-sig')

    print(f'已保存买入记录到: {records_file}')
    print(f'已更新持仓到: {positions_file}')
    return new_records_df


if __name__ == '__main__':
    run_buy()
