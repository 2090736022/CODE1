# 底部策略近一年回测

运行：

```powershell
python botton_year_backtest.py
```

默认以不晚于今天的最新 `daily_basic` 缓存日期为终点，向前回测一个自然年。
也可以固定日期，确保结果可以重复：

```powershell
python botton_year_backtest.py --start 20251015 --end 20261014
```

数据读取自 `model_artifacts/download_daily/daily` 和 `daily_basic`，
股票池来自 `config.STOCK_FILE`；不联网、不修改源策略或股票池。
需要 pandas、numpy 以及源策略已有的依赖。

程序按整个股票池逐日调用 `lianghua1_search_botton.match_strategy`。
不会执行原 `run_search` 中硬编码 `000811.SZ` 的调试语句。
预筛只减少计算量，最终技术条件仍以源策略函数为准；
每个信号日的历史起点与原 `run_search` 相同，防止 MACD 初始化口径改变。

收益统计沿用 `monthly_strategy_backtest.py`：
取信号后 `config.BACKTEST_FORWARD_DAYS` 根日 K（当前为 10），
遍历开盘买入日与严格更晚的收盘卖出日，取最大卖价/买价。
1.1 表示事后最佳收益为 10%。这不是可执行的账户回测，不计算年化收益或账户回撤。
不扣交易费用；原字段里的 `5d` 只是旧名称，实际窗口以参数为准。

默认输出到 `model_artifacts/botton_year_backtest/起点_终点/`：

- `report.html`：详细中文报告，浏览器打开，可以打印成 PDF。
- `report.md`：文本版报告。
- `signals.csv`：所有通过技术和基础条件的信号，含未完成未来窗口的记录。
- `trades.csv`：有完整未来窗口的最佳买卖价、日期和收益。
- `daily_summary.csv`、`monthly_summary.csv`：日度、月度统计。
- `strategy_backtest.csv`：与原月度案例相同的汇总列，收益比值保留三位小数。
- `audit.csv`、`calendar_audit.csv`、`candidate_issues.csv`：覆盖及缺失数据审计。
- `metadata.json`：参数和程序、策略 SHA256。

原案例按有样本日期等权计算月平均，不是所有交易样本直接平均。
无样本日期保持为空；没有足够未来记录的信号不混入收益平均。
停牌股票按照它自己的日 K 记录向前计数，与源函数一致。

可选参数：`--cache-dir`、`--stock-file`、`--output-dir`、`--forward-days`。
`--max-stocks` 只用于小范围调试，不应在正式全池回测时设置。
同一个输出目录再次运行会覆盖该目录中的同名结果文件；原始行情文件不变。

验证：

```powershell
python -m unittest test_botton_year_backtest -v
```

注意当前股票池、当前 ST 名称存在历史状态及幸存者偏差；程序不另做复权。
报告中的最佳收益已使用未来价格选择交易日期，不能视为实盘获利能力。
