from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from lianghua1_search import create_pro
from progress import show_progress


def load_csv_or_empty(file_path, columns):
    """读取 CSV；文件不存在或为空时返回指定列的空 DataFrame。"""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)

    return pd.read_csv(file_path)


def get_daily_data(pro, ts_code, buy_date, end_date):
    """拉取某只股票从买入日到截止日的日 K。"""
    df = pro.daily(ts_code=ts_code, start_date=str(buy_date), end_date=end_date)

    if df is None or df.empty:
        return pd.DataFrame()

    return df.sort_values('trade_date').reset_index(drop=True)


def is_limit_up(row):
    """判断是否涨停。这里用 pct_chg >= 9.8 作为近似口径。"""
    return row['pct_chg'] >= config.LIMIT_UP_PCT


def is_price_yang(current_day, prev_day):
    """阳线定义：今日收盘价高于昨日收盘价。"""
    return current_day['close'] > prev_day['close']


def is_price_yin(current_day, prev_day):
    """阴线定义：今日收盘价低于昨日收盘价。"""
    return current_day['close'] < prev_day['close']


def is_volume_up(current_day, prev_day):
    """放量定义：今日成交量高于昨日成交量。"""
    return current_day['vol'] > prev_day['vol']


def is_shrink_limit_up(current_day, prev_day):
    """缩量涨停：今日涨停，且今日成交量低于昨日成交量。"""
    return is_limit_up(current_day) and current_day['vol'] < prev_day['vol']


def has_upper_shadow_over_body(row):
    """判断是否为上影线大于实体的阳线。实体 = close - open，上影线 = high - close。"""
    body = row['close'] - row['open']
    upper_shadow = row['high'] - row['close']
    return body > 0 and upper_shadow > body


def judge_sell_signal(pro, position, end_date):
    """
    对单条持仓判断最早发生的卖点或买点更新。
    所有卖点按交易日顺序逐日判断，避免未来更晚的卖点抢先返回。
    """
    ts_code = position['ts_code']
    buy_date = str(position['buy_date'])
    buy_price = float(position['buy_price'])
    active_buy_date = buy_date
    active_buy_price = buy_price

    df = get_daily_data(pro, ts_code, buy_date, end_date)

    if len(df) < 2:
        return None

    full_df = df.reset_index(drop=True)
    has_profit = False
    has_loss = False
    update_records = []

    for index in range(1, len(full_df)):
        current_day = full_df.iloc[index]
        prev_day = full_df.iloc[index - 1]

        if current_day['trade_date'] <= active_buy_date:
            continue

        # 卖点 1：出现放量阴线，按当天收盘价卖出。
        if is_price_yin(current_day, prev_day) and is_volume_up(current_day, prev_day):
            return {
                'action': 'sell',
                'effective_buy_date': active_buy_date,
                'effective_buy_price': active_buy_price,
                'updates': update_records,
                'sell_type': '放量阴线卖出',
                'sell_date': current_day['trade_date'],
                'sell_price': current_day['close'],
                'reason': '出现放量阴线，按收盘价卖出',
            }

        # 卖点 2：前一天是缩量涨停板，今天以开盘价卖出。
        if index >= 2 and prev_day['trade_date'] > active_buy_date:
            prev_prev_day = full_df.iloc[index - 2]

            if is_shrink_limit_up(prev_day, prev_prev_day):
                return {
                    'action': 'sell',
                    'effective_buy_date': active_buy_date,
                    'effective_buy_price': active_buy_price,
                    'updates': update_records,
                    'sell_type': '缩量涨停次日卖出',
                    'sell_date': current_day['trade_date'],
                    'sell_price': current_day['open'],
                    'reason': '缩量涨停板的第二天以开盘价卖出',
                }

        if index >= 2:
            prev_prev_day = full_df.iloc[index - 2]
            prev_day_is_yang = is_price_yang(prev_day, prev_prev_day)

            # 卖点 3：买入后上涨，前一天阳线后今天低开超过 1%，按开盘价卖出。
            low_open_over_1 = current_day['open'] < prev_day['close'] * (1 - config.LOW_OPEN_SELL_PCT)

            if has_profit and prev_day_is_yang and low_open_over_1:
                return {
                    'action': 'sell',
                    'effective_buy_date': active_buy_date,
                    'effective_buy_price': active_buy_price,
                    'updates': update_records,
                    'sell_type': '止盈卖出',
                    'sell_date': current_day['trade_date'],
                    'sell_price': current_day['open'],
                    'reason': '买入后上涨，前一天阳线后低开超过1%，按开盘价卖出',
                }

            # 卖点 4：前一天是长上影阳线，今天低开超过 0.5%，按开盘价卖出。
            prev_day_has_long_upper_shadow = prev_day_is_yang and has_upper_shadow_over_body(prev_day)
            low_open_over_05 = (
                current_day['open'] < prev_day['close'] * (1 - config.LONG_UPPER_SHADOW_LOW_OPEN_SELL_PCT)
            )

            if has_profit and prev_day_has_long_upper_shadow and low_open_over_05:
                return {
                    'action': 'sell',
                    'effective_buy_date': active_buy_date,
                    'effective_buy_price': active_buy_price,
                    'updates': update_records,
                    'sell_type': '长上影阳线次日卖出',
                    'sell_date': current_day['trade_date'],
                    'sell_price': current_day['open'],
                    'reason': '前一天阳线且上影线大于实体阳线，第二天低开超过0.5%，按开盘价卖出',
                }

        # 卖点 5：买入后下跌，第一根反弹阳线缩量，按收盘价卖出；放量则更新买点。
        if has_loss and is_price_yang(current_day, prev_day):
            is_shrink_volume = current_day['vol'] < prev_day['vol']

            if is_shrink_volume:
                return {
                    'action': 'sell',
                    'effective_buy_date': active_buy_date,
                    'effective_buy_price': active_buy_price,
                    'updates': update_records,
                    'sell_type': '止损卖出',
                    'sell_date': current_day['trade_date'],
                    'sell_price': current_day['close'],
                    'reason': '买入后下跌，反弹第一根缩量阳线，按收盘价卖出',
                }

            update_records.append({
                'old_buy_date': active_buy_date,
                'old_buy_price': active_buy_price,
                'new_buy_date': current_day['trade_date'],
                'new_buy_price': current_day['close'],
                'reason': '买入后下跌，反弹第一根阳线放量，更新为新买点',
            })
            active_buy_date = current_day['trade_date']
            active_buy_price = current_day['close']
            has_profit = False
            has_loss = False
            continue

        # 当天没有触发卖点时，再更新“买入后是否上涨/下跌”的状态，供后续交易日使用。
        if current_day['high'] > active_buy_price:
            has_profit = True

        if current_day['low'] < active_buy_price:
            has_loss = True

    if update_records:
        last_update = update_records[-1]
        return {
            'action': 'update_buy',
            'new_buy_date': last_update['new_buy_date'],
            'new_buy_price': last_update['new_buy_price'],
            'reason': last_update['reason'],
            'updates': update_records,
        }

    return None


def run_sell(end_date=None, result_dir=None, pro=None):
    """执行卖点判断，更新 持仓.csv，并写入 卖出记录.csv。"""
    pro = pro or create_pro()
    end_date = end_date or config.get_end_date()
    result_dir = Path(result_dir or config.RESULT_DIR)
    positions_file = result_dir / '持仓.csv'
    sell_records_file = result_dir / '卖出记录.csv'
    record_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    position_columns = ['ts_code', 'buy_date', 'buy_price']
    sell_record_columns = [
        'record_time',
        'ts_code',
        'buy_date',
        'buy_price',
        'sell_date',
        'sell_price',
        'sell_type',
        'reason',
    ]

    positions_df = load_csv_or_empty(positions_file, position_columns)
    sell_records_df = load_csv_or_empty(sell_records_file, sell_record_columns)

    remaining_positions = []
    new_sell_records = []

    total_positions = len(positions_df)
    show_progress(0, total_positions, 'sell 判断卖点')

    for index, (_, position) in enumerate(positions_df.iterrows(), start=1):
        ts_code = position['ts_code']

        try:
            signal = judge_sell_signal(pro, position, end_date)

            if signal is None:
                remaining_positions.append(position.to_dict())
                print(f'{ts_code} 暂无卖点')
                continue

            if signal['action'] == 'sell':
                new_sell_records.append({
                    'record_time': record_time,
                    'ts_code': ts_code,
                    'buy_date': signal.get('effective_buy_date', position['buy_date']),
                    'buy_price': signal.get('effective_buy_price', position['buy_price']),
                    'sell_date': signal['sell_date'],
                    'sell_price': signal['sell_price'],
                    'sell_type': signal['sell_type'],
                    'reason': signal['reason'],
                })
                print(f'{ts_code} {signal["sell_type"]}: {signal["sell_date"]}, 价格: {signal["sell_price"]}')
                continue

            if signal['action'] == 'update_buy':
                updated_position = position.to_dict()
                updated_position['buy_date'] = signal['new_buy_date']
                updated_position['buy_price'] = signal['new_buy_price']
                remaining_positions.append(updated_position)
                print(f'{ts_code} 更新买点: {signal["new_buy_date"]}, 价格: {signal["new_buy_price"]}')
                continue

        except Exception as e:
            remaining_positions.append(position.to_dict())
            print(f'{ts_code} 卖点判断失败: {e}')
        finally:
            show_progress(index, total_positions, 'sell 判断卖点')

    result_dir.mkdir(exist_ok=True)

    new_sell_records_df = pd.DataFrame(new_sell_records, columns=sell_record_columns)
    all_sell_records_df = pd.concat([sell_records_df, new_sell_records_df], ignore_index=True)
    all_sell_records_df = all_sell_records_df.drop_duplicates(
        subset=['ts_code', 'buy_date', 'sell_date', 'sell_type'],
        keep='first'
    )
    all_sell_records_df.to_csv(sell_records_file, index=False, encoding='utf-8-sig')

    remaining_positions_df = pd.DataFrame(remaining_positions, columns=position_columns)
    remaining_positions_df = remaining_positions_df.drop_duplicates(subset=['ts_code'], keep='last')
    remaining_positions_df.to_csv(positions_file, index=False, encoding='utf-8-sig')

    print(f'已更新持仓: {positions_file}')
    print(f'已保存卖出记录: {sell_records_file}')
    return new_sell_records_df


if __name__ == '__main__':
    run_sell()
