import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
try:
	import requests
except ImportError:  # pragma: no cover
	requests = None

try:
	import akshare as ak
except ImportError:  # pragma: no cover - optional runtime dependency
	ak = None

import config
from minute_data import get_window, get_cumulative_window, get_full_day
from lianghua1_search import DAILY_CACHE_DIR, create_pro, get_daily_data as get_cached_daily_data
from progress import show_progress
from stock_filters import filter_non_st_stocks


BASE_DIR = Path(__file__).resolve().parent
DAILY_BASIC_CACHE_DIR = BASE_DIR / "model_artifacts" / "download_daily" / "daily_basic"
DAILY_LOOKBACK_YEARS = 1
MIN_DAILY_ROWS = 20
MIN_CIRC_MV_50B = 500_000
MAX_CIRC_MV_1000B = 10_000_000
MIN_CIRC_MV_200B = 2_000_000
MIN_TURNOVER_RATE = 4
MAX_TURNOVER_RATE = 40

# AkShare 东方财富接口免费使用；仅在本地日线缓存缺失时调用。
MINUTE_START = getattr(config, "CHAODUAN_MINUTE_START", "09:45:00")
MINUTE_END = getattr(config, "CHAODUAN_MINUTE_END", "10:30:00")
MINUTE_VOLUME_INCREASE_RATIO = getattr(config, "CHAODUAN_MINUTE_VOLUME_RATIO", 4.0)  # 增长至少 300%


def read_daily_basic_cache(trade_date):
	cache_file = DAILY_BASIC_CACHE_DIR / f"{trade_date}.csv"
	if not cache_file.exists():
		return None
	try:
		return pd.read_csv(cache_file, dtype={"ts_code": str})
	except Exception as error:
		print(f"{cache_file} 读取失败: {error}")
		return None


def get_daily_basic(pro, trade_date):
	"""读取基础数据；旧缓存缺少流通市值时补充下载所需字段。"""
	cached = read_daily_basic_cache(trade_date)
	required_columns = {"ts_code", "turnover_rate_f", "circ_mv"}
	if cached is not None and required_columns.issubset(cached.columns):
		return cached

	fields = "ts_code,trade_date,turnover_rate,turnover_rate_f,circ_mv,total_mv"
	basic_df = pro.daily_basic(trade_date=str(trade_date), fields=fields)
	if basic_df is None or basic_df.empty:
		return pd.DataFrame()

	DAILY_BASIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	basic_df.to_csv(
		DAILY_BASIC_CACHE_DIR / f"{trade_date}.csv",
		index=False,
		encoding="utf-8-sig",
	)
	return basic_df


def _akshare_symbol(ts_code):
	return str(ts_code).split(".")[0].zfill(6)


def _normalise_akshare_daily(df, ts_code):
	"""把 AkShare 中文字段转换为策略使用的 Tushare 字段。"""
	if df is None or df.empty:
		return pd.DataFrame()
	mapping = {
		"日期": "trade_date", "开盘": "open", "最高": "high", "最低": "low",
		"收盘": "close", "成交量": "vol", "成交额": "amount", "涨跌幅": "pct_chg",
		"涨跌额": "change", "换手率": "turnover_rate",
	}
	result = df.rename(columns=mapping).copy()
	if "trade_date" not in result:
		return pd.DataFrame()
	result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
	result["ts_code"] = ts_code
	for column in ("open", "high", "low", "close", "vol", "amount", "pct_chg", "change", "turnover_rate"):
		if column in result:
			result[column] = pd.to_numeric(result[column], errors="coerce")
	return result.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)


def get_daily_data(pro, ts_code, start_date, end_date):
	"""日 K 数据：本地缓存优先，缺失时使用免费 AkShare，最后回退 Tushare。"""
	try:
		return get_cached_daily_data(pro, ts_code, start_date, end_date)
	except (FileNotFoundError, ValueError):
		pass

	result = pd.DataFrame()
	if ak is not None:
		try:
			result = _normalise_akshare_daily(
				ak.stock_zh_a_hist(
					symbol=_akshare_symbol(ts_code), period="daily",
					start_date=str(start_date), end_date=str(end_date), adjust=""
				), ts_code
			)
		except Exception as error:
			print(f"{ts_code} AkShare 日线获取失败，回退 Tushare: {error}")
	if result.empty:
		try:
			fields = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
			result = pro.daily(ts_code=ts_code, start_date=str(start_date), end_date=str(end_date), fields=fields)
			result = result.sort_values("trade_date").reset_index(drop=True) if result is not None else pd.DataFrame()
		except Exception as error:
			raise ValueError(f"{ts_code} 日线数据获取失败（AkShare/Tushare）: {error}") from error
	if result.empty:
		raise ValueError(f"{ts_code} 没有可用日线数据")
	DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	result.to_csv(DAILY_CACHE_DIR / f"{str(ts_code).replace('.', '_')}.csv", index=False, encoding="utf-8-sig")
	return result, True


def _minute_volume(df, trade_date=None):
	if df is None or df.empty:
		return 0.0
	time_column = next((c for c in ("时间", "日期", "datetime", "time") if c in df.columns), None)
	volume_column = next((c for c in ("成交量", "volume", "vol") if c in df.columns), None)
	if time_column is None or volume_column is None:
		return 0.0
	times = pd.to_datetime(df[time_column], errors="coerce")
	clock = times.dt.strftime("%H:%M:%S")
	mask = (clock >= MINUTE_START) & (clock <= MINUTE_END)
	if trade_date is not None:
		mask &= times.dt.strftime("%Y%m%d") == pd.to_datetime(str(trade_date)).strftime("%Y%m%d")
	return float(pd.to_numeric(df.loc[mask, volume_column], errors="coerce").fillna(0).sum())


def get_minute_volume(ts_code, trade_date):
	"""读取新浪真实一分钟 K 线；缺失数据抛错，不冒充零成交量。"""
	return float(get_window(ts_code, trade_date, MINUTE_START, MINUTE_END)["volume"].sum())


def _legacy_get_minute_volume(ts_code, trade_date):
	date = pd.to_datetime(str(trade_date)).strftime("%Y-%m-%d")
	# AkShare 的封装接口偶尔会被东财主动断开，先尝试直连同一免费数据源。
	if requests is not None:
		try:
			market = "1" if str(ts_code).upper().endswith(".SH") else "0"
			code = _akshare_symbol(ts_code)
			params = {
				"secid": f"{market}.{code}", "klt": "1", "fqt": "1",
				"beg": date.replace("-", ""), "end": date.replace("-", ""),
				"fields1": "f1", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
			}
			for host in ("push2his.eastmoney.com", "push2.eastmoney.com"):
				url = f"https://{host}/api/qt/stock/kline/get"
				try:
					response = requests.get(url, params=params, headers={
						"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"
					}, timeout=12)
					response.raise_for_status()
					rows = (response.json().get("data") or {}).get("klines") or []
					if rows:
						values = []
						for row in rows:
							parts = str(row).split(",")
							if len(parts) >= 6 and parts[0][:10] == date and MINUTE_START <= parts[0][-8:] <= MINUTE_END:
								values.append(float(parts[5]))
						return float(sum(values))
				except Exception:
					continue
			raise RuntimeError("东财两个行情域名均无法连接")
		except Exception as error:
			print(f"{ts_code} {trade_date} 东财直连接口失败，尝试 AkShare: {error}")
	if ak is not None:
		for attempt in range(2):
			try:
				df = ak.stock_zh_a_hist_min_em(
					symbol=_akshare_symbol(ts_code), period="1",
					start_date=f"{date} {MINUTE_START}", end_date=f"{date} {MINUTE_END}", adjust=""
				)
				return _minute_volume(df, trade_date)
			except Exception as error:
				if attempt == 1:
					print(f"{ts_code} {trade_date} 一分钟数据获取失败: {error}")
				time.sleep(1.5 * (attempt + 1))
	return 0.0


def get_next_trade_date(pro, selection_date):
	"""按交易所日历获取选股日的下一交易日，跳过周末及休市日。"""
	start = pd.to_datetime(str(selection_date))
	calendar = pro.trade_cal(
		exchange="SSE", start_date=start.strftime("%Y%m%d"),
		end_date=(start + pd.Timedelta(days=40)).strftime("%Y%m%d"),
		is_open="1", fields="cal_date,is_open",
	)
	if calendar is None or calendar.empty:
		raise ValueError(f"{selection_date} 没有可用交易日历，无法确定下一交易日")
	dates = pd.to_datetime(calendar.loc[
		pd.to_numeric(calendar["is_open"], errors="coerce") == 1, "cal_date"
	].astype(str), errors="coerce").dropna()
	if not dates.eq(start).any():
		raise ValueError(f"选股日 {selection_date} 不是交易日")
	following = dates[dates > start]
	if following.empty:
		raise ValueError(f"交易日历中没有 {selection_date} 的下一交易日")
	return following.min().strftime("%Y%m%d")


def observation_volume_ratio(daily, observation_date):
	"""判定日全日成交量 / 含判定日及前四个交易日的五日均量。"""
	data = daily.copy()
	data["trade_date"] = pd.to_datetime(data["trade_date"].astype(str), errors="coerce").dt.strftime("%Y%m%d")
	data = data.dropna(subset=["trade_date"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")
	current = data.loc[data["trade_date"] == observation_date, "vol"]
	window_volumes = pd.to_numeric(data.loc[data["trade_date"] <= observation_date, "vol"].tail(5), errors="coerce")
	volume = pd.to_numeric(current.iloc[-1], errors="coerce") if not current.empty else float("nan")
	if len(window_volumes) != 5 or not window_volumes.map(lambda v: 0 <= v < float("inf")).all() or not 0 <= volume < float("inf") or window_volumes.mean() <= 0:
		raise ValueError(f"{observation_date} 缺少有效全日成交量或含当日的五日均量")
	return float(volume / window_volumes.mean())


def minute_volume_threshold(daily_ratio):
	if daily_ratio > getattr(config, "CHAODUAN_RELAX_DAILY_VOLUME_RATIO", 2.0):
		return getattr(config, "CHAODUAN_RELAXED_MINUTE_VOLUME_RATIO", 3.4)
	return MINUTE_VOLUME_INCREASE_RATIO


def minute_volume_matches(ts_code, observation_date, selection_date, previous_close=None, volume_ratio=None):
	"""同一分钟累计量达到指定倍数，并同时满足下方价格条件。"""
	volume_ratio = MINUTE_VOLUME_INCREASE_RATIO if volume_ratio is None else volume_ratio
	if previous_close is None:
		previous_close = get_cumulative_window(ts_code, selection_date, "15:00:00", "15:00:00").iloc[0]["price"]
	previous_close = float(previous_close)
	if not 0 < previous_close < float("inf"):
		raise ValueError("选股日收盘价无效")
	def series(date):
		frame = get_cumulative_window(ts_code, date, MINUTE_START, MINUTE_END)
		return frame.set_index(pd.to_datetime(frame["datetime"]).dt.strftime("%H:%M:%S")).sort_index()
	observed = series(observation_date)
	selected = series(selection_date)
	if not observed.index.equals(selected.index):
		raise ValueError("两天累计成交量的分钟时刻不一致")
	volume = observed["cumulative_volume"]
	base_volume = selected["cumulative_volume"]
	# 比较同比增幅的百分点差；不是单分钟成交量环比，也不是同比增幅本身。
	# 先扫描完整窗口，已出现买入信号的股票也可能被后续异常跳升排除。
	if (base_volume <= 0).any():
		raise ValueError("选股日窗口内累计成交量为零，无法验证同比增幅跳升")
	growth = (volume / base_volume - 1) * 100
	jump = growth.diff()
	threshold = getattr(config, "CHAODUAN_MAX_VOLUME_GROWTH_JUMP_PP", 200.0)
	# 容忍浮点运算误差，恰好 200 个百分点不剔除。
	abnormal = jump > threshold + 1e-9
	if abnormal.any():
		clock = abnormal[abnormal].index[0]
		position = growth.index.get_loc(clock)
		print(f"跳过 {ts_code}: {growth.index[position - 1]} → {clock} "
			f"累计量同比增幅 {growth.iloc[position - 1]:.4f}% → {growth.loc[clock]:.4f}%，"
			f"跳升 {jump.loc[clock]:.4f} 个百分点，超过 {threshold:g}")
		return False, float(volume.loc[clock]), float(base_volume.loc[clock]), clock, float((observed.loc[clock, "price"] / previous_close - 1) * 100)
	# 比较价格门槛，避免 9.9 / 10 的浮点误差使恰好 -1% 意外入选。
	hits = (base_volume > 0) & (volume >= base_volume * volume_ratio) & (observed["price"] > previous_close * 1)
	matched = bool(hits.any())
	clock = hits[hits].index[0] if matched else observed.index[-1]
	return matched, float(volume.loc[clock]), float(base_volume.loc[clock]), clock, float((observed.loc[clock, "price"] / previous_close - 1) * 100)


def max_volume_minute_is_bullish(ts_code, observation_date, previous_close=None):
	"""最大量分钟收盘不低于开盘，或最高价等于涨停价；并列时每根都须满足。"""
	# 全天 238 根：上午 120 根、下午连续交易 117 根、15:00 集合竞价 1 根。
	# 14:58、14:59 不单独成线；volume 是单分钟量，不是累计量。
	bars = get_full_day(ts_code, observation_date)
	values = bars[["open", "high", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
	for column in ("open", "high", "close", "volume"):
		valid = values[column].map(lambda value: 0 <= value < float("inf"))
		if column != "volume":
			valid &= values[column] > 0
		if not valid.all():
			raise ValueError(f"{observation_date} 分钟 K 线 {column} 数据无效")
	maximum = values["volume"].max()
	if pd.isna(maximum) or maximum <= 0:
		raise ValueError(f"{observation_date} 缺少有效单分钟成交量")
	leaders = values.loc[values["volume"] == maximum]
	allowed = leaders["close"] >= leaders["open"]
	if previous_close is not None:
		previous_close = Decimal(str(previous_close))
		if not previous_close.is_finite() or previous_close <= 0:
			raise ValueError("昨收无效，无法判断最大量分钟最高价是否等于涨停价")
		limit_price = (previous_close * Decimal("1.10")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
		allowed |= leaders["high"].map(lambda price: Decimal(str(price)) == limit_price)
	return bool(allowed.all())


def is_main_board(ts_code):
	code = str(ts_code).split(".")[0].zfill(6)
	market = str(ts_code).upper().split(".")[-1]
	if market == "SZ":
		return code.startswith(("000", "001", "002"))
	if market == "SH":
		return code.startswith(("600", "601", "603", "605"))
	return False


def is_limit_up(row):
	pct_chg = pd.to_numeric(row.get("pct_chg"), errors="coerce")
	if pd.notna(pct_chg):
		return pct_chg >= config.LIMIT_UP_PCT

	close = pd.to_numeric(row.get("close"), errors="coerce")
	pre_close = pd.to_numeric(row.get("pre_close"), errors="coerce")
	if pd.isna(close) or pd.isna(pre_close) or pre_close <= 0:
		return False
	return close >= pre_close * (1 + config.LIMIT_UP_PCT / 100)


def has_three_limit_ups_in_last_year(df):
	data = df.sort_values("trade_date").copy()
	data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
	data = data.dropna(subset=["trade_date"])
	if data.empty:
		return False

	end_date = data["trade_date"].iloc[-1]
	start_date = end_date - pd.DateOffset(years=DAILY_LOOKBACK_YEARS)
	year_data = data[data["trade_date"].between(start_date, end_date)]
	return int(year_data.apply(is_limit_up, axis=1).sum()) >= 3


def match_strategy(df):
	"""匹配涨停历史、当日走势、量能和均线条件。"""
	if df is None or len(df) < MIN_DAILY_ROWS:
		return False

	data = df.sort_values("trade_date").reset_index(drop=True).copy()
	numeric_columns = ["close", "pct_chg", "vol"]
	for column in numeric_columns:
		data[column] = pd.to_numeric(data[column], errors="coerce")
	if data[numeric_columns].tail(MIN_DAILY_ROWS).isna().any().any():
		return False

	last_day = data.iloc[-1]
	previous_day = data.iloc[-2]
	ma5 = data["close"].rolling(5).mean().iloc[-1]
	ma10 = data["close"].rolling(10).mean().iloc[-1]
	ma20 = data["close"].rolling(20).mean().iloc[-1]

	return (
		has_three_limit_ups_in_last_year(data)
		and last_day["pct_chg"] < 1
		and last_day["vol"] <= previous_day["vol"]
		and (ma5 > ma10 or ma10 > ma20)
	)


def basic_matches(row, ts_code):
	turnover = pd.to_numeric(row.get("turnover_rate_f"), errors="coerce")
	if pd.isna(turnover):
		turnover = pd.to_numeric(row.get("turnover_rate"), errors="coerce")
	circ_mv = pd.to_numeric(row.get("circ_mv"), errors="coerce")
	if pd.isna(turnover) or pd.isna(circ_mv):
		return False
	if not MIN_TURNOVER_RATE < turnover < MAX_TURNOVER_RATE:
		return False

	code = str(ts_code).upper()
	if code.endswith(".SZ"):
		return MIN_CIRC_MV_50B <= circ_mv <= MAX_CIRC_MV_1000B
	if code.endswith(".SH"):
		return circ_mv > MIN_CIRC_MV_200B
	return False


def run_search(
	as_of_date=None,
	result_dir=None,
	stock_file=None,
	pro=None,
	request_sleep=0,
	save_file=True,
	*,
	daily_matcher=None,
	basic_matcher=None,
	minute_prefilter=None,
	observation_min_turnover=None,
	observation_positive_close=False,
	observation_min_volume_ratio=None,
	result_prefix="超短筛选结果",
):
	"""截至配置日期进行日 K 选股，再用下一交易日早盘量对比选股日。"""
	pro = pro or create_pro()
	daily_matcher = daily_matcher or match_strategy
	basic_matcher = basic_matcher or basic_matches
	as_of_date = pd.to_datetime(str(as_of_date or config.get_end_date())).strftime("%Y%m%d")
	observation_date = get_next_trade_date(pro, as_of_date)
	observation_turnover = None
	now = pd.Timestamp.now(tz="Asia/Shanghai")
	if now < pd.Timestamp(observation_date + " 15:00:00", tz="Asia/Shanghai"):
		raise ValueError(f"{observation_date} 尚未收盘，无法判断收盘条件")
	if observation_min_turnover is not None:
		observation_basic = get_daily_basic(pro, observation_date)
		if observation_basic.empty or not {"ts_code", "turnover_rate_f"}.issubset(observation_basic.columns):
			raise ValueError(f"{observation_date} 缺少收盘实际换手率数据，请待数据发布后重试")
		observation_basic = observation_basic.copy()
		observation_basic["ts_code"] = observation_basic["ts_code"].astype(str).str.strip()
		observation_turnover = pd.to_numeric(observation_basic.drop_duplicates("ts_code").set_index("ts_code")["turnover_rate_f"], errors="coerce")
	print(f"日 K 选股日: {as_of_date}；分钟线观察日: {observation_date}（对比选股日同时段）")
	result_dir = Path(result_dir or config.RESULT_DIR)
	stock_file = Path(stock_file or config.STOCK_FILE)
	start_date = (
		pd.to_datetime(as_of_date) - pd.DateOffset(years=DAILY_LOOKBACK_YEARS)
	).strftime("%Y%m%d")

	stock_data = filter_non_st_stocks(pd.read_csv(stock_file, dtype={"ts_code": str}))
	codes = [code.strip() for code in stock_data["ts_code"].dropna().astype(str)]
	codes = [code for code in codes if is_main_board(code)]
	basic_data = get_daily_basic(pro, as_of_date)
	if basic_data.empty or "ts_code" not in basic_data.columns:
		raise ValueError(f"{as_of_date} 没有可用 daily_basic 数据")
	basic_data["ts_code"] = basic_data["ts_code"].astype(str).str.strip()
	basic_by_code = basic_data.drop_duplicates("ts_code").set_index("ts_code")

	result = []
	show_progress(0, len(codes), "chaoduan 筛选")
	for index, ts_code in enumerate(codes, start=1):
		try:
			if ts_code not in basic_by_code.index or not basic_matcher(
				basic_by_code.loc[ts_code], ts_code
			):
				continue

			if observation_turnover is not None:
				turnover = observation_turnover.get(ts_code, float("nan"))
				if not observation_min_turnover < turnover < float("inf"):
					continue

			daily_df, downloaded = get_daily_data(pro, ts_code, start_date, as_of_date)
			daily_df = daily_df.copy()
			daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"].astype(str), errors="coerce").dt.strftime("%Y%m%d")
			daily_df = daily_df[daily_df["trade_date"].between(start_date, as_of_date)]
			daily_df = daily_df.sort_values("trade_date").reset_index(drop=True)
			if daily_df.empty or daily_df.iloc[-1]["trade_date"] != as_of_date:
				print(f"{ts_code} 缺少选股日 {as_of_date} 日 K，跳过")
				continue
			if daily_matcher(daily_df):
				observation_daily, _ = get_daily_data(pro, ts_code, start_date, observation_date)
				daily_ratio = observation_volume_ratio(observation_daily, observation_date)
				minute_ratio = minute_volume_threshold(daily_ratio)
				if observation_min_volume_ratio is not None and daily_ratio <= observation_min_volume_ratio:
					continue
				if observation_positive_close:
					closing_df, _ = get_daily_data(pro, ts_code, observation_date, observation_date)
					closing_dates = pd.to_datetime(closing_df["trade_date"].astype(str), errors="coerce").dt.strftime("%Y%m%d")
					closing_rows = closing_df.loc[closing_dates == observation_date]
					if closing_rows.empty:
						raise ValueError(f"{observation_date} 缺少收盘日 K 数据")
					closing_row = closing_rows.iloc[-1]
					closing_pct = pd.to_numeric(closing_row.get("pct_chg"), errors="coerce")
					if pd.isna(closing_pct):
						close = pd.to_numeric(closing_row.get("close"), errors="coerce")
						pre_close = pd.to_numeric(closing_row.get("pre_close"), errors="coerce")
						if pd.isna(pre_close) or pre_close <= 0:
							raise ValueError(f"{observation_date} 缺少有效涨幅和昨收数据")
						closing_pct = (close / pre_close - 1) * 100
					if not 0 < closing_pct < float("inf"):
						continue
				if minute_prefilter is not None and not minute_prefilter(
					ts_code, observation_date, daily_df.iloc[-1]["close"],
					observation_close=observation_daily.loc[
						pd.to_datetime(observation_daily["trade_date"].astype(str)).dt.strftime("%Y%m%d") == observation_date,
						"close"
					].iloc[-1],
					daily_volume_ratio=daily_ratio,
				):
					continue
				minute_match, today_volume, yesterday_volume, match_time, minute_pct_chg = minute_volume_matches(
					ts_code, observation_date, as_of_date, previous_close=daily_df.iloc[-1]["close"],
					volume_ratio=minute_ratio
				)
				if not minute_match:
					continue
				if not max_volume_minute_is_bullish(ts_code, observation_date, previous_close=daily_df.iloc[-1]["close"]):
					continue
				row = basic_by_code.loc[ts_code]
				result.append(
					{
						"ts_code": ts_code,
						"trade_date": as_of_date,
						"minute_trade_date": observation_date,
						"previous_trade_date": as_of_date,
						"minute_match_time": match_time,
						"minute_pct_chg": minute_pct_chg,
						"cumulative_volume": today_volume,
						"previous_cumulative_volume": yesterday_volume,
						"minute_volume_ratio": today_volume / yesterday_volume,
						"observation_daily_volume_ratio": daily_ratio,
						"minute_volume_threshold": minute_ratio,
						"volume_increase_pct": (today_volume / yesterday_volume - 1) * 100,
						"volume_unit": "手",
						"volume_source": "腾讯当日累计成交量",
						"circ_mv": row.get("circ_mv"),
						"turnover_rate_f": row.get("turnover_rate_f", row.get("turnover_rate")),
					}
				)
				print(
					f"\n符合条件: {ts_code} | 首次达标时刻: {match_time} | 当时涨幅: {minute_pct_chg:.4f}%\n"
					f"  观察日 {observation_date} 截至 {match_time} 当日累计成交量: {today_volume:,.2f} 手\n"
					f"  选股日 {as_of_date} 截至 {match_time} 当日累计成交量: {yesterday_volume:,.2f} 手\n"
					f"  倍数: {today_volume / yesterday_volume:.6f} | "
					f"增长: {(today_volume / yesterday_volume - 1) * 100:.4f}% | "
					f"门槛: >= {minute_ratio:g} 倍 | 全日量比: {daily_ratio:.4f} | 数据源: 腾讯当日累计成交量"
				)
				if observation_turnover is not None:
					result[-1]["observation_turnover_rate_f"] = float(observation_turnover.loc[ts_code])
					print(f"  观察日 {observation_date} 收盘实际换手率: {observation_turnover.loc[ts_code]:.4f}%")
			if downloaded and request_sleep:
				time.sleep(request_sleep)
		except Exception as error:
			print(f"{ts_code} 获取失败: {error}")
		finally:
			show_progress(index, len(codes), "chaoduan 筛选")

	result_df = pd.DataFrame(
		result,
		columns=["ts_code", "trade_date", "minute_trade_date", "previous_trade_date",
			"minute_match_time", "minute_pct_chg", "cumulative_volume", "previous_cumulative_volume",
			"minute_volume_ratio", "observation_daily_volume_ratio", "minute_volume_threshold", "volume_increase_pct", "volume_unit", "volume_source", "circ_mv", "turnover_rate_f"]
	)
	if observation_turnover is not None:
		result_df["observation_turnover_rate_f"] = [row["observation_turnover_rate_f"] for row in result]
	if save_file:
		result_dir.mkdir(parents=True, exist_ok=True)
		output_file = result_dir / f"{result_prefix}_{as_of_date}.csv"
		result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
		print(f"已保存筛选结果: {output_file}")
	return result_df


if __name__ == "__main__":
	run_search()
