from datetime import datetime
from pathlib import Path


# =====================
# 基础路径
# =====================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / '结果'
STOCK_FILE = BASE_DIR / 'number.csv'


# =====================
# 日期配置
# =====================

# DAILY_RUN_DATE 为空时使用当天日期；如需复盘指定日期，改成 '20260527' 这种格式。
DAILY_RUN_DATE = ''

# 买入、卖出、删除判断的截止日期。为空时默认等于 DAILY_RUN_DATE。
END_DATE = '20261013'

def today_str():
    """返回 YYYYMMDD 格式的当天日期。"""
    return datetime.today().strftime('%Y%m%d')


def get_run_date():
    """获取本次日常分析使用的日期。"""
    return DAILY_RUN_DATE or today_str()


def get_end_date():
    """获取买入、卖出、删除判断使用的截止日期。"""
    return END_DATE or get_run_date()


# =====================
# Tushare 配置
# =====================

TUSHARE_TOKEN = ''


# =====================
# 运行开关
# =====================

RUN_SEARCH = True
RUN_STORE = True
RUN_BUY = True
RUN_SELL = True
RUN_DELETE = True

# 日常运行通常只判断当天筛选结果的买点；如果要扫描结果文件夹里所有筛选结果，改成 False。
BUY_ONLY_TODAY_SEARCH = True

# 接口请求间隔，遇到权限限制时可以调大。
SEARCH_REQUEST_SLEEP = 0.0
BUY_REQUEST_SLEEP = 0.0

# 控制台进度条。
SHOW_PROGRESS = True
PROGRESS_BAR_WIDTH = 30
PROGRESS_PRINT_EVERY = 20


# =====================
# 筛选参数
# =====================

# daily_basic 的 total_mv 单位是“万元”。700000 万元 = 70 亿元。
MIN_TOTAL_MV = 1000000
MIN_TURNOVER_RATE = 10
LIMIT_UP_PCT = 9.51
LIMIT_STREAK_LOOKBACK_DAYS = 20
MIN_LIMIT_UP_STREAK = 2
MAX_LIMIT_UP_STREAK = 5
SEARCH_DAILY_LOOKBACK_YEARS = 1
MIN_DAILY_ROWS_FOR_SEARCH = 120
RECENT_VOLUME_WINDOW = 10
MIN_LAST_VOLUME_PEAK_RATIO = 0.5
MA_DAYS_FOR_SEARCH = 15
BACKTEST_FORWARD_DAYS = 10


# =====================
# 买入参数
# =====================
#1111

BUY_LOOKBACK_DAYS_BEFORE_SEARCH = 90
MONTH_LOOKBACK_DAYS = 22
MAX_MONTH_GAIN_BEFORE_BUY = 1.0


# =====================
# 卖出参数
# =====================

LOW_OPEN_SELL_PCT = 0.01
LONG_UPPER_SHADOW_LOW_OPEN_SELL_PCT = 0.005


# =====================
# 删除股票池参数
# =====================

DELETE_DAILY_LOOKBACK_YEARS = 1
DELETE_MA_DAYS = 15
DELETE_CONFIRM_DAYS = 3

# chaoduan：get_end_date() 为选股日；下一交易日 09:45–10:30 逐分钟比较
# 截至同一时刻的当日累计成交量（含开盘集合竞价），增长至少 300%，即 >= 4 倍。
CHAODUAN_MINUTE_START = '09:45:00'
CHAODUAN_MINUTE_END = '10:30:00'
CHAODUAN_MINUTE_VOLUME_RATIO = 4.0
# 窗口内相邻分钟累计量同比增幅向上跳升超过此百分点数，整只股票排除。
CHAODUAN_MAX_VOLUME_GROWTH_JUMP_PP = 200.0

# 判定日全日量 / 含当日及前四个交易日的五日均量严格大于 2 时，早盘增长门槛降为 240%。
CHAODUAN_RELAX_DAILY_VOLUME_RATIO = 2.0
CHAODUAN_RELAXED_MINUTE_VOLUME_RATIO = 3.4
