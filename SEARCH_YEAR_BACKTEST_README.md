# test 和 turn 策略年度回测

在 D:\pythonp 终端中分别运行：

```powershell
python test_year_backtest.py
python turn_year_backtest.py
```

默认以不晚于今天的最新本地 daily_basic 日期为终点，向前回测一年。
指定固定区间：

```powershell
python test_year_backtest.py --start 20251015 --end 20261014
python turn_year_backtest.py --start 20251015 --end 20261014
```

结果分别位于 `model_artifacts/test_year_backtest/日期区间/` 和
`model_artifacts/turn_year_backtest/日期区间/`。
双击各目录中的 report.html 查看详细中文报告，浏览器打印可另存为 PDF。
report.md 是文本版；signals.csv 是所有信号；trades.csv 是完整未来窗口的最佳买卖明细。
daily_summary.csv、monthly_summary.csv 和 strategy_backtest.csv 给出汇总，
audit.csv、calendar_audit.csv、candidate_issues.csv、metadata.json 用于核对覆盖和参数。

两个入口共用 botton_year_backtest.py 的读取、统计和报告模块，
year_backtest_strategies.py 提供各自策略适配；这些文件需要放在同一个目录。
原策略文件不修改、不联网、不写入原结果股票池。

test 入口包含原 run_search 的 match_strategy 和 match_volume_price_strategy 两个分支，
branch 列分别为 main、volume_price。沿用两个独立 if 的输出口径，若同日同股同时命中两分支则保留两条。
turn 使用其原 match_strategy；基础条件也调用各自原 check_basic_conditions，
不会套用底部策略的换手率、市值和 MACD 规则。

收益沿用 monthly_strategy_backtest.py：信号后 10 根日 K（可用 --forward-days 改变）内，
选择先开盘买入、严格更晚的收盘卖出的最大价格比值。
这是事后最佳收益统计，不是可执行收益或账户年度收益；不扣费用，也不模拟资金、涨跌停成交和重叠持仓。
日均按当日完整样本等权，月均按有完整样本日期的日均等权；缺失未来数据的信号单独保留，不混入均值。

参数还支持 --cache-dir、--stock-file、--output-dir、--max-stocks（仅调试用）。
同一个输出目录再次运行会覆盖同名结果，原始行情文件不变。

运行校验：

```powershell
python -m unittest test_botton_year_backtest test_search_year_backtests -v
```

当前股票池和 ST 名称存在历史状态及幸存者偏差；日 K 不另做复权。
策略源代码发生修改后，应重新运行校验，尤其注意预筛与原函数的一致性。
