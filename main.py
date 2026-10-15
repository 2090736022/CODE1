import csv
from pathlib import Path

import config
from lianghua1_buy import run_buy
from lianghua1_delete import run_delete
from lianghua1_search import create_pro, run_search
from lianghua1_sell import run_sell


def log(message):
    """统一打印主程序运行进度。"""
    print(f'[main] {message}')


def build_pool_from_search_results(result_dir):
    """汇总结果目录下所有筛选结果，生成去重后的 result.csv 股票池。"""
    result_dir = Path(result_dir)
    output_file = result_dir / 'result.csv'
    seen_codes = set()
    result_codes = []

    for file_path in sorted(result_dir.glob('量化筛选结果*.csv')):
        with file_path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames or 'ts_code' not in reader.fieldnames:
                log(f'跳过 {file_path.name}: 没有 ts_code 列')
                continue

            for row in reader:
                ts_code = row.get('ts_code', '').strip()

                if not ts_code or ts_code in seen_codes:
                    continue

                seen_codes.add(ts_code)
                result_codes.append(ts_code)

    result_dir.mkdir(exist_ok=True)

    with output_file.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ts_code'])
        writer.writerows([[ts_code] for ts_code in result_codes])

    log(f'已更新股票池: {output_file}，数量: {len(result_codes)}')
    return output_file


def main():
    """日常量化分析主程序。"""
    run_date = config.get_run_date()
    end_date = config.get_end_date()
    result_dir = Path(config.RESULT_DIR)
    stock_file = Path(config.STOCK_FILE)

    result_dir.mkdir(exist_ok=True)

    log(f'运行日期: {run_date}')
    log(f'判断截止日期: {end_date}')
    log(f'结果目录: {result_dir}')
    log(f'股票池文件: {stock_file}')

    pro = create_pro()

    if config.RUN_SEARCH:
        log('开始执行 search 筛选')
        search_df = run_search(
            as_of_date=run_date,
            result_dir=result_dir,
            stock_file=stock_file,
            pro=pro,
            request_sleep=config.SEARCH_REQUEST_SLEEP,
            save_file=True,
        )
        log(f'search 完成，本次筛选数量: {len(search_df)}')

    if config.RUN_STORE:
        log('开始汇总筛选结果到 result.csv')
        build_pool_from_search_results(result_dir)

    if config.RUN_BUY:
        log('开始执行 buy 买点判断')
        only_search_date = run_date if config.BUY_ONLY_TODAY_SEARCH else None
        buy_df = run_buy(
            end_date=end_date,
            result_dir=result_dir,
            only_search_date=only_search_date,
            pro=pro,
            request_sleep=config.BUY_REQUEST_SLEEP,
        )
        log(f'buy 完成，本次新增买点数量: {len(buy_df)}')

    if config.RUN_SELL:
        log('开始执行 sell 卖点判断')
        sell_df = run_sell(
            end_date=end_date,
            result_dir=result_dir,
            pro=pro,
        )
        log(f'sell 完成，本次卖出数量: {len(sell_df)}')

    if config.RUN_DELETE:
        log('开始执行 delete 股票池删除判断')
        delete_df = run_delete(
            end_date=end_date,
            result_dir=result_dir,
            pro=pro,
        )
        log(f'delete 完成，本次删除数量: {len(delete_df)}')

    log('日常量化分析完成')


if __name__ == '__main__':
    main()
