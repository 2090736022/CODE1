from datetime import datetime
from pathlib import Path

import pandas as pd
import tushare as ts

import config
from progress import show_progress


# 兼容其他模块旧导入；真实参数统一在 config.py 中修改。
TOKEN = config.TUSHARE_TOKEN
MIN_TOTAL_MV = config.MIN_TOTAL_MV
BASE_DIR = Path(__file__).resolve().parent
SEARCH_CACHE_DIR = BASE_DIR / 'model_artifacts' / 'download_daily'
DAILY_CACHE_DIR = SEARCH_CACHE_DIR / 'daily'
DAILY_BASIC_CACHE_DIR = SEARCH_CACHE_DIR / 'daily_basic'
DAILY_FIELDS = 'ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
DAILY_BASIC_FIELDS = 'ts_code,trade_date,turnover_rate,total_mv'
DOWNLOAD_YEARS = 5

def log(message):
    """统一打印主程序运行进度。"""
    print(f'[main] {message}')

def create_pro():
    """创建 Tushare pro 接口对象。"""
    ts.set_token(config.TUSHARE_TOKEN)
    return ts.pro_api()


def ensure_cache_dirs():
    DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_BASIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_code(ts_code):
    return str(ts_code).replace('.', '_')


def five_year_start_date(end_date):
    return (pd.to_datetime(str(end_date)) - pd.DateOffset(years=DOWNLOAD_YEARS)).strftime('%Y%m%d')


def read_cached_csv(cache_file):
    cache_file = Path(cache_file)
    if not cache_file.exists():
        return None
    try:
        return pd.read_csv(cache_file, dtype={'ts_code': str, 'trade_date': str})
    except Exception as error:
        print(f'{cache_file} 读取失败，将重新下载: {error}')
        return None


def get_daily_data(pro, ts_code, start_date, end_date):
    """优先读取本地日线缓存；缺缓存时一次性下载截至 end_date 的五年数据。"""
    ensure_cache_dirs()
    cache_file = DAILY_CACHE_DIR / f'{cache_code(ts_code)}.csv'
    cached = read_cached_csv(cache_file)
    if cached is None or cached.empty:
        raise FileNotFoundError(f'{ts_code} 没有本地日线缓存，请先运行 python download.py')

    cached['trade_date'] = cached['trade_date'].astype(str)
    cached = cached.drop_duplicates(subset=['trade_date'], keep='last')
    cached = cached.sort_values('trade_date').reset_index(drop=True)
    result = cached[(cached['trade_date'] >= str(start_date)) & (cached['trade_date'] <= str(end_date))].copy()
    if result.empty:
        raise ValueError(f'{ts_code} 本地日线缓存不覆盖 {start_date}-{end_date}，请先运行 python download.py')
    return result, False


def is_limit_up(row):
    """判断单日是否涨停。这里用 pct_chg >= 9.51 作为近似口径。"""
    return row['pct_chg'] >= config.LIMIT_UP_PCT


def has_limit_up_streak_between_3_and_5(df):
    """判断最近 20 个交易日内是否出现 3 到 5 连涨停。"""
    recent_df = df.tail(config.LIMIT_STREAK_LOOKBACK_DAYS).reset_index(drop=True)
    #print(df)
    if len(recent_df) < config.MIN_LIMIT_UP_STREAK:
        return False

    def row_is_limit_up(row):
        pct_chg = pd.to_numeric(row.get('pct_chg'), errors='coerce')

        if pd.notna(pct_chg) and pct_chg >= config.LIMIT_UP_PCT:
            return True

        close = pd.to_numeric(row.get('close'), errors='coerce')
        pre_close = pd.to_numeric(row.get('pre_close'), errors='coerce')

        if pd.isna(close) or pd.isna(pre_close) or pre_close <= 0:
            return False

        return close >= pre_close * (1 + config.LIMIT_UP_PCT / 100)

    current_streak = 0

    for _, row in recent_df.iterrows():
        #print(row)
        if row_is_limit_up(row):
            current_streak += 1
            
            continue

        if config.MIN_LIMIT_UP_STREAK <= current_streak <= config.MAX_LIMIT_UP_STREAK:
            return True

        current_streak = 0

    return config.MIN_LIMIT_UP_STREAK <= current_streak <= config.MAX_LIMIT_UP_STREAK


def match_strategy(df):
    """日线筛选条件。传入某只股票截至筛选日的日 K 数据。"""
    if df is None or len(df) < config.MIN_DAILY_ROWS_FOR_SEARCH:
        return False

    df = df.sort_values('trade_date').reset_index(drop=True)

    last_day = df.iloc[-1]
    prev_day = df.iloc[-2]
    prev_prev_day = df.iloc[-3]

    has_limit_up_streak = has_limit_up_streak_between_3_and_5(df)
    cc=prev_prev_day['close']>prev_day['close']
    #print(last_day['close'])
    #last_not_limit_up = not is_limit_up(last_day)
    last_is_yang = last_day['close'] / prev_day['close'] - 1 > -0.0025
    last_volume_up = last_day['vol'] > prev_day['vol']
    last_volume_not_too_low = (
        last_day['vol'] >= df.tail(config.RECENT_VOLUME_WINDOW)['vol'].max() * config.MIN_LAST_VOLUME_PEAK_RATIO
    )
    prev_is_yin = (prev_prev_day['close'] > prev_day['close']) or (prev_day['open']>prev_day['close'])
    prev_volume_down = prev_day['vol'] < prev_prev_day['vol']
    last_close_above_ma = last_day['close'] > df.tail(config.MA_DAYS_FOR_SEARCH)['close'].mean()
    if last_day['pct_chg'] >9.80:
        last_volume_up = True
        last_volume_not_too_low = True

    #print(has_limit_up_streak,     last_is_yang, last_volume_up,prev_is_yin,prev_volume_down,last_close_above_ma,cc)
    return (
        has_limit_up_streak
        and last_is_yang
        and last_volume_up
        and last_volume_not_too_low
        and prev_is_yin
        and prev_volume_down
        and last_close_above_ma
        and cc
    )

def build_pool_from_search_results(result_dir):
    """汇总结果目录下所有筛选结果，生成去重后的 result.csv 股票池。"""
    result_df = result_dir
    result_dir = Path(config.RESULT_DIR)
    output_file = result_dir / 'result.csv'
    seen_codes = set()
    result_rows = []

    result_dir.mkdir(exist_ok=True)

    if output_file.exists():
        try:
            old_result_df = pd.read_csv(output_file)
        except Exception as e:
            print(f'{output_file} 读取失败: {e}')
            old_result_df = pd.DataFrame(columns=['ts_code', 'trade_date'])

        if 'ts_code' in old_result_df.columns:
            for _, row in old_result_df.iterrows():
                ts_code = str(row.get('ts_code', '')).strip()
                if ts_code and ts_code not in seen_codes:
                    seen_codes.add(ts_code)
                    result_rows.append({
                        'ts_code': ts_code,
                        'trade_date': row.get('trade_date', ''),
                    })

    if isinstance(result_df, pd.DataFrame) and 'ts_code' in result_df.columns:
        for _, row in result_df.iterrows():
            ts_code = str(row.get('ts_code', '')).strip()
            if ts_code and ts_code not in seen_codes:
                seen_codes.add(ts_code)
                result_rows.append({
                    'ts_code': ts_code,
                    'trade_date': row.get('trade_date', ''),
                })

    pool_df = pd.DataFrame(result_rows, columns=['ts_code', 'trade_date'])
    pool_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'已更新股票池: {output_file}')
    return pool_df

def get_daily_basic(pro, ts_code, trade_date):
    """一次性拉取筛选日的换手率和总市值。"""
    ensure_cache_dirs()
    cache_file = DAILY_BASIC_CACHE_DIR / f'{trade_date}.csv'
    cached = read_cached_csv(cache_file)
    if cached is None or cached.empty:
        raise FileNotFoundError(f'{trade_date} 没有本地 daily_basic 缓存，请先运行 python download.py')
    return cached[cached['ts_code'].astype(str) == str(ts_code)].copy()


def check_basic_conditions(pro, ts_code, trade_date):
    """筛选日基本面条件：换手率大于 10%，总市值大于设定阈值。"""
    basic_df = get_daily_basic(pro, ts_code, trade_date)

    if basic_df is None or basic_df.empty:
        return False

    row = basic_df.iloc[0]
    turnover_over_10 = row['turnover_rate'] > config.MIN_TURNOVER_RATE
    total_mv_over_limit = row['total_mv'] > config.MIN_TOTAL_MV

    return turnover_over_10 and total_mv_over_limit


def check_turnover_over_10(pro, ts_code, trade_date):
    """兼容旧调用：只判断换手率是否大于 10%。"""
    basic_df = get_daily_basic(pro, ts_code, trade_date)

    if basic_df is None or basic_df.empty:
        return False

    return basic_df.iloc[0]['turnover_rate'] > config.MIN_TURNOVER_RATE


def run_search(
    as_of_date=None,
    result_dir=None,
    stock_file=None,
    pro=None,
    request_sleep=0.08,
    save_file=True,
):
    """执行股票筛选。"""
    pro = pro or create_pro()
    as_of_date = as_of_date or config.get_end_date()
    print(as_of_date)
    result_dir = Path(result_dir or config.RESULT_DIR)
    stock_file = Path(stock_file or config.STOCK_FILE)
    start_date = str(int(as_of_date) - config.SEARCH_DAILY_LOOKBACK_YEARS * 10000)
    #print(start_date)

    number_data = pd.read_csv(stock_file)

    if 'ts_code' not in number_data.columns:
        raise ValueError('number.csv 中没有 ts_code 列')

    codes = number_data['ts_code'].dropna().astype(str).str.strip().tolist()
    result = []
    total_codes = len(codes)
    show_progress(0, total_codes, 'search 计算中')

    for index, ts_code in enumerate(codes, start=1):
        try:
            #ts_code='002119.SZ'
            df, _ = get_daily_data(pro, ts_code, start_date, as_of_date)
            if df is None or df.empty:
                continue
            df = df.sort_values('trade_date').reset_index(drop=True)
            #print(df)
            last_trade_date = df.iloc[-1]['trade_date']
          
            #print(match_strategy(df))
            if match_strategy(df) and check_basic_conditions(pro, ts_code, last_trade_date):
                print(1)
                result.append({
                    'ts_code': ts_code,
                    'trade_date': last_trade_date,
                })
                print(f'符合条件: {ts_code}')

            #print(result)

        except Exception as e:
            print(f'{ts_code} 获取失败: {e}')
        finally:
            show_progress(index, total_codes, 'search 计算中')

    result_df = pd.DataFrame(result, columns=['ts_code', 'trade_date'])
    print(result_df )

    if save_file:
        result_dir.mkdir(exist_ok=True)
        output_file = result_dir / f'量化筛选结果{as_of_date}.csv'
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f'已保存筛选结果: {output_file}')

        build_pool_from_search_results(result_df)

    return result_df

if __name__ == '__main__':
    run_search()
