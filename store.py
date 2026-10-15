import csv
from pathlib import Path


# store.py 用于汇总“结果”文件夹中的所有筛选结果。
# 输入：结果/量化筛选结果YYYYMMDD.csv
# 输出：结果/result.csv

result_dir = Path(__file__).resolve().parent / '结果'
output_file = result_dir / 'result.csv'

seen_codes = set()
result_codes = []

# 遍历所有筛选结果文件，把 ts_code 合并到一个股票池中。
for file_path in sorted(result_dir.glob('量化筛选结果*.csv')):
    with file_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or 'ts_code' not in reader.fieldnames:
            print(f'跳过 {file_path.name}: 没有 ts_code 列')
            continue

        for row in reader:
            ts_code = row.get('ts_code', '').strip()

            # 股票代码为空或已经出现过时跳过，避免重复进入股票池。
            if not ts_code or ts_code in seen_codes:
                continue

            seen_codes.add(ts_code)
            result_codes.append(ts_code)

result_dir.mkdir(exist_ok=True)

# 写出最终股票池。
with output_file.open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ts_code'])

    for ts_code in result_codes:
        writer.writerow([ts_code])

print(f'已保存 {len(result_codes)} 个股票代码到 {output_file}')
