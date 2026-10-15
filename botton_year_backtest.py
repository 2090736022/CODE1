"""本地底部策略年度回测。收益口径与 monthly_strategy_backtest.py 一致。

python botton_year_backtest.py
python botton_year_backtest.py --start 20251015 --end 20261014

报告为 UTF-8 HTML（浏览器打开、打印为 PDF）和 Markdown；明细为 CSV。
不调用 run_search，避免其硬编码股票、重复扫描和修改股票池。
"""
import argparse
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
import lianghua1_search_botton as strategy
from stock_filters import filter_non_st_stocks

ROOT = Path(__file__).resolve().parent
RETURN = "future_5d_high_close_to_low_open_return"
DETAIL_COLUMNS = ["ts_code", "name", "as_of_date", "month", "turnover_rate", "total_mv",
                  "status", "future_rows", "future_5d_lowest_open", "future_5d_lowest_open_date",
                  "future_5d_highest_close", "future_5d_highest_close_date", RETURN, "return_pct"]


def best_forward_result(future, days):
    """复现原函数：先买后卖，卖出日期严格晚于买入日期；没有交易费用。"""
    if len(future) < days:
        return None
    future = future.iloc[:days].copy()
    for col in ("open", "close"):
        future[col] = pd.to_numeric(future[col], errors="coerce")
    future = future.dropna(subset=["open", "close"]).reset_index(drop=True)
    best = None
    for i in range(len(future) - 1):
        buy = future.iloc[i]
        if buy["open"] <= 0:
            continue
        sell = future.loc[future.iloc[i + 1:]["close"].idxmax()]
        ratio = sell["close"] / buy["open"]
        if best is None or ratio > best[RETURN]:
            best = {"future_5d_lowest_open": buy["open"],
                    "future_5d_lowest_open_date": buy["trade_date"],
                    "future_5d_highest_close": sell["close"],
                    "future_5d_highest_close_date": sell["trade_date"], RETURN: ratio,
                    "return_pct": (ratio - 1) * 100}
    return best


def candidate_mask(df):
    """仅用原策略必要条件预筛；最终仍调用原 match_strategy。"""
    c, v = df["close"], df["vol"]
    return ((df["pct_chg"] > 5) & (df["pct_chg"].shift() <= .1)
            & (v >= v.shift()) & (v > v.shift().rolling(20).mean() * 1.05)
            & (c > c.rolling(21).mean()) & (c > c.rolling(120).mean())
            & (df["high"] > c.rolling(5).mean()) & (df["high"] > c.rolling(10).mean())
            & (c.rolling(60).mean() > c.rolling(60).mean().shift())
            & (c.rolling(120).mean() > c.rolling(120).mean().shift())
            & (c.rolling(60).mean() >= c.rolling(120).mean() * 1.01))


def summarize(detail, calendar, start, end):
    valid = detail[detail["status"] == "complete"].copy()
    daily = pd.DataFrame({"as_of_date": calendar})
    daily["month"] = daily["as_of_date"].str[:6]
    counts = detail.groupby("as_of_date").size()
    daily["signal_count"] = daily["as_of_date"].map(counts).fillna(0).astype(int)
    agg = valid.groupby("as_of_date")[RETURN].agg(["mean", "count"])
    daily["average_return"] = daily["as_of_date"].map(agg["mean"])
    daily["complete_count"] = daily["as_of_date"].map(agg["count"]).fillna(0).astype(int)
    daily["average_return_pct"] = (daily["average_return"] - 1) * 100
    # 与案例一样：月均为有样本日期的日均收益比值的算术平均。
    monthly = daily.groupby("month", as_index=False).agg(
        trading_days=("as_of_date", "size"), signal_count=("signal_count", "sum"),
        complete_count=("complete_count", "sum"), signal_days=("complete_count", lambda s: int((s > 0).sum())),
        average_return=("average_return", "mean"))
    monthly = pd.DataFrame({"month": pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M").strftime("%Y%m")}).merge(monthly, how="left", on="month")
    for col in ("trading_days", "signal_count", "complete_count", "signal_days"):
        monthly[col] = monthly[col].fillna(0).astype(int)
    monthly["average_return_pct"] = (monthly["average_return"] - 1) * 100
    return valid, daily, monthly


def fmt(value, digits=2):
    return "无数据" if pd.isna(value) else f"{value:.{digits}f}"


def monthly_chart(monthly):
    """无外部依赖的 SVG 月度收益图，空样本月份不画柱子。"""
    maximum = max(5, monthly["average_return_pct"].abs().max())
    if pd.isna(maximum):
        maximum = 5
    width, baseline, scale = 1050, 200, 150 / maximum
    svg = [f'<svg viewBox="0 0 {width} 420" role="img" aria-label="各月份事后最佳收益率均值柱状图" xmlns="http://www.w3.org/2000/svg">',
           '<rect width="1050" height="420" fill="#fff"/>',
           '<text x="25" y="28" font-size="18">各月份日均最佳收益率的平均值（%）</text>',
           f'<line x1="30" y1="{baseline}" x2="1030" y2="{baseline}" stroke="#a0acba"/>']
    for n, (_, row) in enumerate(monthly.iterrows()):
        x = 40 + n * 76
        value = row["average_return_pct"]
        if pd.notna(value):
            height = abs(value) * scale
            y = baseline-height if value >= 0 else baseline
            svg += [f'<rect x="{x}" y="{y:.2f}" width="42" height="{height:.2f}" fill="#27799c"/>',
                    f'<text x="{x}" y="{y-8:.2f}" font-size="13">{value:.1f}%</text>']
        else:
            svg += [f'<text x="{x}" y="185" font-size="12">无样本</text>']
        svg += [f'<text x="{x}" y="375" font-size="12">{row["month"]}</text>']
    svg.append('</svg>')
    return "\n".join(svg)


def write_report(out, meta, detail, daily, monthly, audit):
    valid = detail[detail["status"] == "complete"]
    r = pd.to_numeric(valid["return_pct"], errors="coerce")
    mean_daily = daily["average_return_pct"].mean()
    mean_monthly = monthly["average_return_pct"].mean()
    lines = ["# 底部策略近一年回测报告", "",
        f"生成时间：{meta['generated_at']}。信号区间：{meta['start']} 至 {meta['end']}。",
        f"本报告复现 monthly_strategy_backtest.py 的事后最佳收益口径，未来窗口为 {meta['forward_days']} 个股票交易记录。",
        "",
        "## 主要结果", "",
        f"扫描股票 {meta['stock_count']} 只，成功读取 {meta['loaded_stocks']} 只；日历覆盖 {len(daily)} 个交易日。",
        f"共发现 {len(detail)} 条股票日期信号，其中 {len(valid)} 条可计算收益，{len(detail)-len(valid)} 条未完成或无有效未来价格；涉及 {detail['ts_code'].nunique()} 只股票。",
        f"有样本日的日均收益率再平均为 {fmt(mean_daily)}%；逐笔等权平均为 {fmt(r.mean())}%；有样本月份的月均再平均为 {fmt(mean_monthly)}%。三者权重不同。",
        "该收益来自事后选择最佳买卖日期，不代表可执行收益、年度账户收益或复利收益。", "",
        "## 回测和统计口径", "",
        "1. 每个交易日按当日及以前的数据判断信号；每个判断日的数据起点与原 run_search 一致，为日期数值减去配置的回看年数乘 10000，MACD 也在该截断区间重新计算。",
        f"2. 信号日不参与收益计算。取其后 {meta['forward_days']} 根日 K，遍历所有开盘买入日，再在严格晚于买入日的日期中选择最高收盘价卖出，保留最大的卖价/买价。买入不一定在次日，也不一定是窗口最低开盘价。",
        "3. 收益比值 1.100 表示收益率 10%。保留旧列名 future_5d 以便对照，但实际窗口由 forward_days 决定。",
        "4. 日均为当日所有完整样本收益比值的平均；月均为该月有完整样本的日均比值的平均。无样本日记为空，不按零收益混入平均。",
        "5. 与案例一致，未扣佣金、印花税、滑点，不模拟涨跌停成交、资金规模、仓位、复利或重叠持仓。不会输出账户净值、最大回撤或夏普率。",
        "6. 股票停牌时，未来窗口按该股票实际日 K 记录推进；即使超过市场的同数量交易日，也遵循原函数口径。",
        "7. 数据只读取本地 CSV，不联网、不补下载，不更新原股票池。未来窗口可使用信号区间结束后的已缓存数据，但信号计算不会使用未来数据。", "",
        "## 筛选条件", "",
        f"所有条件同时满足。至少 {max(121, config.MIN_DAILY_ROWS_FOR_SEARCH)} 根日 K；最近 120 根 high、close、pre_close、pct_chg、vol 无缺失。",
        f"普通换手率 turnover_rate > 4%，总市值 total_mv > {config.MIN_TOTAL_MV} 万元，即 > {config.MIN_TOTAL_MV/10000:g} 亿元。",
        "当日涨幅 > 5%，昨日涨幅 ≤ 0.1%；当日成交量 ≥ 昨日成交量，且 > 前 20 日均量的 1.05 倍（均量不含当日）。",
        f"此前 3 个交易日无涨停。涨停判断沿用原函数，pct_chg ≥ {config.LIMIT_UP_PCT}，或收盘价达到昨收价相应倍数。",
        "此前 12 个月内，任意连续 5 个交易日至少两次涨停，或累计至少五次涨停；历史窗口不含当日。",
        "收盘价 > 最近 21 日均价且 > MA120；最高价 > MA5 且 > MA10。MA60、MA120 均比昨日上升，且 MA60 ≥ MA120 × 1.01。",
        "MACD 参数为 12、26、9，柱值为 2×(DIF−DEA)。当日柱值 < 0 且高于昨日，昨日柱值低于前日；当日 DIF > 0、DEA > 0。",
        "股票来自当前 number.csv，名称含 ST 的排除（仅在名称列存在时生效）。绕过原 run_search 中强制指定 000811.SZ 的调试语句，对整个股票池调用原 match_strategy。", "",
        "## 样本收益分布", ""]
    metrics = {"完整样本数": len(r), "平均收益率 %": fmt(r.mean()), "中位数 %": fmt(r.median()),
        "标准差 %": fmt(r.std()), "最低值 %": fmt(r.min()), "最高值 %": fmt(r.max()),
        "25% 分位 %": fmt(r.quantile(.25)), "75% 分位 %": fmt(r.quantile(.75)),
        "最佳收益仍大于零的比例 %": fmt((r > 0).mean()*100 if len(r) else np.nan),
        "最佳收益达到 5% 的比例 %": fmt((r >= 5).mean()*100 if len(r) else np.nan),
        "最佳收益达到 10% 的比例 %": fmt((r >= 10).mean()*100 if len(r) else np.nan)}
    display_cols = ["ts_code", "name", "as_of_date", "future_5d_lowest_open_date",
                    "future_5d_lowest_open", "future_5d_highest_close_date", "future_5d_highest_close", "return_pct"]
    labels = {"ts_code": "代码", "name": "名称", "as_of_date": "信号日", "future_5d_lowest_open_date": "最佳买入日",
              "future_5d_lowest_open": "买入开盘价", "future_5d_highest_close_date": "最佳卖出日",
              "future_5d_highest_close": "卖出收盘价", "return_pct": "最佳收益率 %"}
    month_labels = {"month": "月份", "trading_days": "交易日数", "signal_count": "信号数", "complete_count": "完整样本数",
                    "signal_days": "有完整样本日数", "average_return": "日均比值再平均", "average_return_pct": "对应收益率 %"}
    tables = [("分布统计", pd.DataFrame(metrics.items(), columns=["指标", "数值"])),
              ("逐月统计", monthly.rename(columns=month_labels)),
              ("最佳十条完整样本", valid.sort_values("return_pct", ascending=False).head(10)[display_cols].rename(columns=labels)),
              ("最差十条完整样本", valid.sort_values("return_pct").head(10)[display_cols].rename(columns=labels)),
              ("未完成样本", detail[detail["status"] != "complete"][["ts_code", "name", "as_of_date", "status", "future_rows"]])]
    lines += ["“大于零比例”描述的是事后最佳结果，不是实盘胜率。", "",
        "## 数据覆盖和偏差", "",
        f"本地日 K 文件数 {meta['daily_files']}，回测区间 daily_basic 日期数 {meta['basic_dates']}；股票交易日相对基础缓存日期缺口数 {meta['missing_basic_dates']}。",
        f"候选判断次数 {meta['candidates']}，原策略技术条件通过次数 {meta['technical_matches']}，基础数据缺失次数 {meta['basic_missing']}，历史长度不足次数 {meta['short_history']}。",
        f"有区间日 K 的股票 {int(audit['period_rows'].gt(0).sum())} 只；日 K 最后日期早于信号区间最后缓存交易日的股票 {int((audit['last_date'].fillna('') < daily['as_of_date'].max()).sum())} 只（可能为停牌、退市或缓存未更新，不能仅据此判定原因）。区间首月和末月可能是不完整自然月。",
        "audit.csv 保留逐股票读取状态、最早/最晚数据日期、区间记录数及错误；日期缺口见 calendar_audit.csv。缺失数据不当作策略拒绝，基础数据不足的候选单独记录。",
        "当前股票池和当前 ST 名称用于历史回测，存在幸存者偏差及历史状态偏差；没有历史退市股票池。日线使用缓存原始价格，未另做复权，除权除息可能影响跨日价格比值和指标。",
        "报告不证明每一日所有股票均完整覆盖；仅在本地具备数据的股票日期上判断。最后若干日期通常因未来数据不足而无法评价，应结合 daily_summary.csv 的完整样本数阅读。",
        "MACD 初始化依赖原策略截断窗口。近一年边界和股票上市日期不足以覆盖所需历史时，不产生可验证信号。", "",
        "## 输出文件和复现", "",
        "signals.csv：全部通过条件的信号及未来数据状态。trades.csv：可计算的最佳买卖明细。daily_summary.csv：逐交易日统计。monthly_summary.csv：逐月统计。",
        "strategy_backtest.csv：与原案例同列的每日和月度汇总，比值四舍五入到三位小数。audit.csv：逐股票覆盖情况。candidate_issues.csv：基础数据缺失候选。metadata.json：参数、源文件哈希和数据覆盖。",
        f"运行命令：python botton_year_backtest.py --start {meta['start']} --end {meta['end']} --forward-days {meta['forward_days']}。",
        "收益细节未提前四舍五入，汇总以原始精度计算。未把平均最佳收益换算成年化收益。", ""]
    report_title = meta.get("report_title", "底部策略近一年回测报告")
    if "strategy_rules" in meta:
        # 共享统计模板，但条件和运行命令必须来自各自策略。
        a, z = lines.index("## 筛选条件"), lines.index("## 样本收益分布")
        lines[a:z] = ["## 筛选条件", "", *meta["strategy_rules"], ""]
        lines[0] = "# " + report_title
        lines = [line.replace("python botton_year_backtest.py", "python " + meta["entry_script"]) for line in lines]
        if meta.get("strategy") == "test":
            lines = [line.replace("，MACD 也在该截断区间重新计算", "")
                     .replace("MACD 初始化依赖原策略截断窗口。", "指标和历史涨停检查依赖原策略截断窗口。") for line in lines]
        if "branch" in detail:
            branch_stats = detail.groupby("branch", as_index=False).agg(
                signal_count=("status", "size"), complete_count=("status", lambda s: int(s.eq("complete").sum())))
            tables.append(("信号分支统计", branch_stats))
    # HTML 自包含，无 CDN、外部脚本或图片依赖；适合浏览器阅读和打印。
    css = "body{font:15px/1.75 'Microsoft YaHei',sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#243247}h1,h2{color:#153c60}table{border-collapse:collapse;width:100%;font-size:12px;margin:18px 0}th,td{border:1px solid #ccd6df;padding:7px;text-align:left;overflow-wrap:anywhere}th{background:#edf3f8}tr:nth-child(even){background:#f7f9fb}@media print{body{margin:0;max-width:none}table{font-size:9px}thead{display:table-header-group}tr{break-inside:avoid}h2{break-after:avoid}}"
    parts = [f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>{html.escape(report_title)}</title>", f"<style>{css}</style><body>"]
    for line in lines:
        if line.startswith("# "):
            parts.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            parts.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line:
            parts.append(f"<p>{html.escape(line)}</p>")
    for title, frame in tables:
        parts.append(f"<h2>{title}</h2>")
        if title == "逐月统计":
            parts.append(monthly_chart(monthly))
        parts.append(frame.to_html(index=False, na_rep="无数据", float_format=lambda x: f"{x:.4f}", escape=True))
        lines += [f"## {title}", "", frame.to_csv(index=False), ""]
    parts.append("</body></html>")
    (out / "report.html").write_text("\n".join(parts), encoding="utf-8")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args, spec=None):
    source = spec.module if spec is not None else strategy
    strategy_name = spec.name if spec is not None else "botton"
    cache = Path(args.cache_dir)
    daily_dir, basic_dir = cache / "daily", cache / "daily_basic"
    basic_files = {p.stem: p for p in basic_dir.glob("*.csv") if p.stem.isdigit() and len(p.stem) == 8}
    if not basic_files:
        raise ValueError("没有本地 daily_basic 缓存")
    today = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d")
    end = args.end or max(d for d in basic_files if d <= today)
    end_ts = pd.to_datetime(end, format="%Y%m%d")
    start = args.start or ((end_ts - pd.DateOffset(years=1)) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    if pd.to_datetime(start, format="%Y%m%d") > end_ts:
        raise ValueError("start 必须不晚于 end")
    if args.forward_days < 2:
        raise ValueError("forward-days 必须至少为 2")
    out = Path(args.output_dir or ROOT / "model_artifacts" / f"{strategy_name}_year_backtest" / f"{start}_{end}")
    out.mkdir(parents=True, exist_ok=True)
    stocks = filter_non_st_stocks(pd.read_csv(args.stock_file, dtype={"ts_code": str})).drop_duplicates("ts_code")
    if args.max_stocks:
        stocks = stocks.head(args.max_stocks)
    calendar = {d for d in basic_files if start <= d <= end}
    signals, audit, issues = [], [], []
    basics, basic_raws = {}, {}
    counters = dict(candidates=0, technical_matches=0, basic_missing=0, short_history=0)
    for n, (_, stock) in enumerate(stocks.iterrows(), 1):
        code = stock["ts_code"]
        path = daily_dir / (code.replace(".", "_") + ".csv")
        record = {"ts_code": code, "status": "ok", "rows": 0, "period_rows": 0}
        try:
            df = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
            df = source.normalize_daily_frame(df).drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)
            if df.empty:
                raise ValueError("空日 K")
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.strftime("%Y%m%d")
            period = df["trade_date"].between(start, end)
            calendar.update(df.loc[period, "trade_date"])
            record.update(rows=len(df), period_rows=int(period.sum()), first_date=df.iloc[0]["trade_date"], last_date=df.iloc[-1]["trade_date"])
            indices = df.index[period & (spec.candidate_mask(df) if spec is not None else candidate_mask(df))]
            for i in indices:
                counters["candidates"] += 1
                day = df.iloc[i]["trade_date"]
                lower = str(int(day) - config.SEARCH_DAILY_LOOKBACK_YEARS * 10000)
                history = df.iloc[:i+1]
                history = history[history["trade_date"] >= lower].copy()
                if len(history) < (spec.min_rows if spec is not None else max(121, config.MIN_DAILY_ROWS_FOR_SEARCH)):
                    counters["short_history"] += 1
                    continue
                if spec is None:
                    # 用完全相同的历史区间作 MACD 快筛。
                    macd = strategy.add_macd(history.copy())
                    t, y, p = macd.iloc[-1], macd.iloc[-2], macd.iloc[-3]
                    if not (t["macd"] < 0 and t["macd"] > y["macd"] and y["macd"] < p["macd"] and t["diff"] > 0 and t["dea"] > 0):
                        continue
                    branches = ["main"] if strategy.match_strategy(history) else []
                else:
                    branches = spec.match(history)
                if not branches:
                    continue
                counters["technical_matches"] += 1
                if day not in basics:
                    if day not in basic_files:
                        basics[day] = None
                    else:
                        try:
                            b = pd.read_csv(basic_files[day], dtype={"ts_code": str}, usecols=["ts_code", "turnover_rate", "total_mv"])
                            basic_raws[day] = b.copy()
                            b = b.drop_duplicates("ts_code", keep="first").set_index("ts_code")
                            basics[day] = b.apply(pd.to_numeric, errors="coerce")
                        except (ValueError, OSError, pd.errors.ParserError) as exc:
                            basics[day] = None
                            issues.append({"ts_code": "*", "as_of_date": day, "reason": str(exc)})
                b = basics[day]
                if b is None or code not in b.index or b.loc[code].isna().any():
                    counters["basic_missing"] += 1
                    issues.append({"ts_code": code, "as_of_date": day, "reason": "基础数据缺失或无效"})
                    continue
                row = b.loc[code]
                basic_ok = (row["turnover_rate"] > 4 and row["total_mv"] > config.MIN_TOTAL_MV) if spec is None else spec.basic_matches(basic_raws[day], code, day, history)
                if not basic_ok:
                    continue
                future = df.iloc[i+1:i+1+args.forward_days]
                result = best_forward_result(future, args.forward_days)
                for branch in branches:
                    signals.append({"ts_code": code, "name": stock.get("name", ""), "as_of_date": day, "month": day[:6],
                        "turnover_rate": row["turnover_rate"], "total_mv": row["total_mv"], "future_rows": len(future),
                        **({"branch": branch} if spec is not None else {}),
                        "status": "complete" if result else ("insufficient_future" if len(future) < args.forward_days else "invalid_future_prices"), **(result or {})})
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            record.update(status="error", error=str(exc))
        audit.append(record)
        if n % 100 == 0 or n == len(stocks):
            print(f"[backtest] {n}/{len(stocks)} stocks; signals={len(signals)}", flush=True)
    detail = pd.DataFrame(signals, columns=DETAIL_COLUMNS + (["branch"] if spec is not None else [])).sort_values(["as_of_date", "ts_code"]).reset_index(drop=True)
    if not calendar:
        raise ValueError("回测区间没有本地日期")
    calendar = sorted(calendar)
    valid, daily, monthly = summarize(detail, calendar, start, end)
    audit = pd.DataFrame(audit).reindex(columns=["ts_code", "status", "rows", "period_rows", "first_date", "last_date", "error"])
    meta = {"start": start, "end": end, "forward_days": args.forward_days,
        "generated_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "stock_count": len(stocks), "loaded_stocks": int(audit["status"].eq("ok").sum()),
        "daily_files": len(list(daily_dir.glob("*.csv"))), "basic_dates": sum(start <= d <= end for d in basic_files),
        "missing_basic_dates": len(set(calendar) - set(basic_files)), "cache_dir": str(cache.resolve()),
        "stock_file": str(Path(args.stock_file).resolve()), "lookback_years": config.SEARCH_DAILY_LOOKBACK_YEARS,
        "min_total_mv": config.MIN_TOTAL_MV, "limit_up_pct": config.LIMIT_UP_PCT,
        "source_sha256": hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **counters}
    if spec is not None:
        meta.update(strategy=strategy_name, report_title=spec.report_title, strategy_rules=spec.rules,
                    entry_script=f"{strategy_name}_year_backtest.py", min_total_mv=spec.min_total_mv,
                    min_turnover_rate=spec.min_turnover_rate,
                    adapter_sha256=hashlib.sha256(Path(spec.source_file).read_bytes()).hexdigest())
    for name, frame in [("signals", detail), ("trades", valid), ("daily_summary", daily), ("monthly_summary", monthly),
                        ("audit", audit), ("candidate_issues", pd.DataFrame(issues, columns=["ts_code", "as_of_date", "reason"]))]:
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"trade_date": calendar, "has_daily_basic": [d in basic_files for d in calendar]}).to_csv(out / "calendar_audit.csv", index=False, encoding="utf-8-sig")
    compatible = []
    for kind, frame in [("daily_strategy_average", daily), ("monthly_strategy_average", monthly)]:
        f = frame.copy()
        f["row_type"], f["strategy"] = kind, strategy_name
        if "as_of_date" not in f:
            f["as_of_date"] = ""
        compatible.append(f[["row_type", "month", "as_of_date", "strategy", "average_return"]])
    pd.concat(compatible).round({"average_return": 3}).to_csv(out / "strategy_backtest.csv", index=False, encoding="utf-8-sig", float_format="%.3f")
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out, meta, detail, daily, monthly, audit)
    print(json.dumps({"output": str(out.resolve()), "signals": len(detail), "complete": len(valid), "stocks_loaded": meta["loaded_stocks"]}, ensure_ascii=False), flush=True)
    return out


def parse_args(description=None):
    p = argparse.ArgumentParser(description=description or __doc__)
    p.add_argument("--start", help="信号区间起点 YYYYMMDD，默认截至日期向前一年加一天")
    p.add_argument("--end", help="信号区间终点 YYYYMMDD，默认不晚于今日的最新基础缓存日期")
    p.add_argument("--forward-days", type=int, default=config.BACKTEST_FORWARD_DAYS)
    p.add_argument("--cache-dir", default=str(ROOT / "model_artifacts" / "download_daily"))
    p.add_argument("--stock-file", default=str(ROOT / config.STOCK_FILE))
    p.add_argument("--output-dir")
    p.add_argument("--max-stocks", type=int, help="仅用于小样本验证，正式运行不要设置")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
