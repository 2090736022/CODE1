from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from lianghua1_buy import find_buy_with_tushare
from lianghua1_search import create_pro
from progress import show_progress


def load_pool(pool_file):
    """读取股票池 result.csv；没有文件时返回空股票池。"""
    if not pool_file.exists() or pool_file.stat().st_size == 0:
        return pd.DataFrame(columns=['ts_code'])

    return pd.read_csv(pool_file)


def load_buy_records(records_file):
    """读取买入记录；没有文件时返回空 DataFrame。"""
    if not records_file.exists() or records_file.stat().st_size == 0:
        return pd.DataFrame(columns=['buy_date', 'ts_code'])

    return pd.read_csv(records_file)


def parse_yyyymmdd(date_value):
    """把 YYYYMMDD 格式日期转成 datetime；失败时返回 None。"""
    if pd.isna(date_value):
        return None

    date_text = str(date_value).strip()

    if not date_text:
        return None

    try:
        return datetime.strptime(date_text, '%Y%m%d')
    except ValueError:
        return None


def should_delete_no_buy_after_three_days(pro, pool_row, end_date):
    """自放入 result.csv 起满 3 天仍未触发买入，则从股票池删除。"""
    ts_code = str(pool_row.get('ts_code', '')).strip()
    pool_date = str(pool_row.get('trade_date', '')).strip()

    if not ts_code or not pool_date:
        return False, None

    pool_datetime = parse_yyyymmdd(pool_date)
    end_datetime = parse_yyyymmdd(end_date)

    if pool_datetime is None or end_datetime is None:
        return False, None

    if end_datetime < pool_datetime:
        return False, None

    days_in_pool = (end_datetime - pool_datetime).days

    if days_in_pool < 3:
        return False, None

    buy_record = find_buy_with_tushare(pro, ts_code, pool_date, end_date)

    if buy_record is not None:
        return False, None

    return True, {
        'record_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ts_code': ts_code,
        'delete_date': end_date,
        'pool_date': pool_date,
        'days_in_pool': days_in_pool,
        'reason': '自放入result.csv起连续三天没有触发买入',
    }


def should_delete_from_pool(pro, ts_code, end_date):
    """
    删除规则：
    股票已连续三天收盘价低于 30 分钟 K 线 120 均线。

    注意：当前只有日 K 数据，无法精确计算 30 分钟 K 线 120 均线。
    这里沿用近似口径：A 股一天约 8 根 30 分钟 K，120 根约等于 15 个交易日，
    因此用日线 MA15 近似。
    """
    start_date = str(int(end_date) - config.DELETE_DAILY_LOOKBACK_YEARS * 10000)
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    if df is None or len(df) < config.DELETE_MA_DAYS + config.DELETE_CONFIRM_DAYS - 1:
        return False, None

    df = df.sort_values('trade_date').reset_index(drop=True)
    ma_column = f'ma{config.DELETE_MA_DAYS}'
    df[ma_column] = df['close'].rolling(window=config.DELETE_MA_DAYS).mean()

    last_three_days = df.tail(config.DELETE_CONFIRM_DAYS)

    if last_three_days[ma_column].isna().any():
        return False, None

    is_below_ma = (last_three_days['close'] < last_three_days[ma_column]).all()

    if not is_below_ma:
        return False, None

    delete_record = {
        'record_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ts_code': ts_code,
        'delete_date': last_three_days.iloc[-1]['trade_date'],
        'reason': f'连续{config.DELETE_CONFIRM_DAYS}天收盘价低于30分钟K线120均线的日线近似MA{config.DELETE_MA_DAYS}',
    }

    for index, (_, row) in enumerate(last_three_days.iterrows(), start=1):
        delete_record[f'close_{index}'] = row['close']
        delete_record[f'ma{config.DELETE_MA_DAYS}_{index}'] = row[ma_column]

    return True, delete_record


def run_delete(end_date=None, result_dir=None, pro=None):
    """按删除规则更新 result.csv 股票池，并写入 删除记录.csv。"""
    end_date = end_date or config.get_end_date()
    result_dir = Path(result_dir or config.RESULT_DIR)
    pool_file = result_dir / 'result.csv'
    delete_records_file = result_dir / '删除记录.csv'

    pool_df = load_pool(pool_file)

    if 'ts_code' not in pool_df.columns:
        raise ValueError('result.csv 中没有 ts_code 列')

    delete_records = []
    delete_codes = set()

    total_codes = len(pool_df)
    show_progress(0, total_codes, 'delete 判断股票池')

    for index, (_, pool_row) in enumerate(pool_df.iterrows(), start=1):
        ts_code = str(pool_row.get('ts_code', '')).strip()

        if not ts_code:
            continue

        try:
            pro = pro or create_pro()
            need_delete, delete_record = should_delete_no_buy_after_three_days(pro, pool_row, end_date)

            if not need_delete:
                need_delete, delete_record = should_delete_from_pool(pro, ts_code, end_date)

            if need_delete:
                delete_codes.add(ts_code)
                delete_records.append(delete_record)
                print(f'删除股票池: {ts_code}')
            else:
                print(f'保留股票池: {ts_code}')
        except Exception as e:
            print(f'{ts_code} 删除判断失败: {e}')
        finally:
            show_progress(index, total_codes, 'delete 判断股票池')

    result_dir.mkdir(exist_ok=True)

    # 从股票池中删除命中的股票。
    new_pool_df = pool_df[~pool_df['ts_code'].astype(str).isin(delete_codes)].copy()
    new_pool_df.to_csv(pool_file, index=False, encoding='utf-8-sig')

    delete_record_columns = [
        'record_time',
        'ts_code',
        'delete_date',
        'pool_date',
        'days_in_pool',
        'reason',
    ]

    for index in range(1, config.DELETE_CONFIRM_DAYS + 1):
        delete_record_columns.append(f'close_{index}')
        delete_record_columns.append(f'ma{config.DELETE_MA_DAYS}_{index}')

    if delete_records_file.exists() and delete_records_file.stat().st_size > 0:
        old_delete_records_df = pd.read_csv(delete_records_file)
    else:
        old_delete_records_df = pd.DataFrame(columns=delete_record_columns)

    # 删除记录按 ts_code + delete_date 去重。
    new_delete_records_df = pd.DataFrame(delete_records, columns=delete_record_columns)
    all_delete_records_df = pd.concat([old_delete_records_df, new_delete_records_df], ignore_index=True)
    all_delete_records_df = all_delete_records_df.drop_duplicates(subset=['ts_code', 'delete_date'], keep='first')
    all_delete_records_df.to_csv(delete_records_file, index=False, encoding='utf-8-sig')

    print(f'股票池剩余数量: {len(new_pool_df)}')
    print(f'本次删除数量: {len(delete_codes)}')
    print(f'已更新股票池: {pool_file}')
    print(f'已保存删除记录: {delete_records_file}')
    return new_delete_records_df


if __name__ == '__main__':
    run_delete()
