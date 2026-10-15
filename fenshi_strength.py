# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from fenshi import get_minute_em


DATETIME_COLUMNS = ("day", "datetime", "time", "date", "日期时间", "时间", "日期")
OPEN_COLUMNS = ("open", "开盘")
HIGH_COLUMNS = ("high", "最高")
LOW_COLUMNS = ("low", "最低")
CLOSE_COLUMNS = ("close", "收盘", "最新价")
VOLUME_COLUMNS = ("volume", "成交量")
MODEL_DIR = Path(__file__).resolve().parent / "model_artifacts"
MARKET_MODEL_PATH = MODEL_DIR / "fenshi_strength_market.joblib"


@dataclass
class StrengthResult:
    stock_code: str
    trade_date: str
    minute_count: int
    total_volume: float
    up_volume: float
    down_volume: float
    neutral_volume: float
    buy_strength: float
    recent_buy_strength: float
    big_volume_buy_strength: float
    pressure_score: float
    vwap_distance: float
    momentum_recent: float
    volume_acceleration: float
    upper_shadow_ratio: float
    volume_pile_ratio: float
    volume_pile_bias: float
    volume_pile_reaction: float
    post_pile_pressure: float
    close_position: float
    recent_signal_score: float
    rule_score: float
    model_score: float | None
    final_score: float
    latest_close: float
    vwap: float | None
    above_vwap: bool | None


def normalize_stock_code(value: str) -> str:
    stock_code = str(value).strip().lower()
    if stock_code.startswith(("sh", "sz")):
        return stock_code
    if stock_code.startswith("60"):
        return f"sh{stock_code}"
    return stock_code


def model_path(stock_code: str) -> Path:
    safe_code = normalize_stock_code(stock_code).replace(".", "_")
    return MODEL_DIR / f"fenshi_strength_{safe_code}.joblib"


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or pd.isna(denominator):
        return default
    return float(numerator / denominator)


def find_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    label: str,
    required: bool = True,
) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"分时数据缺少{label}字段，实际字段: {list(df.columns)}")
    return None


def normalize_trade_date(value: str) -> str:
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    raise ValueError("日期格式错误，请输入 YYYY-MM-DD 或 YYYYMMDD")


def prepare_minute_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if df is None or df.empty:
        raise ValueError("没有拉取到分时数据")

    data = df.copy()
    columns = {
        "datetime": find_column(data, DATETIME_COLUMNS, "时间"),
        "open": find_column(data, OPEN_COLUMNS, "开盘价"),
        "high": find_column(data, HIGH_COLUMNS, "最高价", required=False),
        "low": find_column(data, LOW_COLUMNS, "最低价", required=False),
        "close": find_column(data, CLOSE_COLUMNS, "收盘价"),
        "volume": find_column(data, VOLUME_COLUMNS, "成交量"),
    }

    data["_minute_datetime"] = pd.to_datetime(data[columns["datetime"]], errors="coerce")
    data = data.dropna(subset=["_minute_datetime"]).copy()
    if data.empty:
        raise ValueError(f"{columns['datetime']} 字段无法解析出有效日期时间")

    data["_trade_date"] = data["_minute_datetime"].dt.strftime("%Y-%m-%d")
    for key in ("open", "high", "low", "close", "volume"):
        column = columns[key]
        if column:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data[columns["volume"]] = data[columns["volume"]].fillna(0)
    data = data.dropna(subset=[columns["open"], columns["close"]]).copy()
    data = data.sort_values("_minute_datetime")

    return data, columns


def get_available_dates(data: pd.DataFrame) -> list[str]:
    return sorted(data["_trade_date"].dropna().unique().tolist())


def calculate_day_features(
    day_data: pd.DataFrame,
    columns: dict[str, str | None],
    recent_window: int = 30,
) -> dict[str, float]:
    open_column = columns["open"]
    high_column = columns["high"]
    low_column = columns["low"]
    close_column = columns["close"]
    volume_column = columns["volume"]

    day_data = day_data.sort_values("_minute_datetime").copy()
    day_data["_prev_close"] = day_data[close_column].shift(1)

    direction_by_k = day_data[close_column] - day_data[open_column]
    direction_by_prev = day_data[close_column] - day_data["_prev_close"]
    up_mask = (direction_by_k > 0) | ((direction_by_k == 0) & (direction_by_prev > 0))
    down_mask = (direction_by_k < 0) | ((direction_by_k == 0) & (direction_by_prev < 0))
    neutral_mask = ~(up_mask | down_mask)

    volume = day_data[volume_column].fillna(0)
    signed_volume = volume.where(up_mask, -volume.where(down_mask, 0))
    total_volume = float(volume.sum())
    up_volume = float(volume[up_mask].sum())
    down_volume = float(volume[down_mask].sum())
    neutral_volume = float(volume[neutral_mask].sum())
    buy_strength = ratio(up_volume + neutral_volume * 0.5, total_volume, 0.5)

    recent = day_data.tail(max(1, recent_window))
    recent_volume = recent[volume_column].fillna(0)
    recent_total_volume = float(recent_volume.sum())
    recent_up_volume = float(recent_volume[up_mask.loc[recent.index]].sum())
    recent_down_volume = float(recent_volume[down_mask.loc[recent.index]].sum())
    recent_buy_strength = ratio(recent_up_volume, recent_up_volume + recent_down_volume, 0.5)

    volume_quantile = volume.quantile(0.75) if len(volume) >= 8 else volume.mean()
    big_mask = volume >= volume_quantile
    big_up_volume = float(volume[big_mask & up_mask].sum())
    big_down_volume = float(volume[big_mask & down_mask].sum())
    big_volume_buy_strength = ratio(big_up_volume, big_up_volume + big_down_volume, 0.5)

    pressure = volume * (day_data[close_column] - day_data[open_column]) / day_data[open_column].replace(0, pd.NA)
    pressure_score = float(pressure.fillna(0).sum()) / total_volume if total_volume > 0 else 0.0

    first_open = float(day_data.iloc[0][open_column])
    latest_close = float(day_data.iloc[-1][close_column])
    previous_close = float(day_data.iloc[-min(len(day_data), recent_window + 1)][close_column])
    day_return = ratio(latest_close - first_open, first_open)
    momentum_recent = ratio(latest_close - previous_close, previous_close)

    if total_volume > 0:
        vwap = float((day_data[close_column] * volume).sum() / total_volume)
        vwap_distance = ratio(latest_close - vwap, vwap)
        above_vwap = 1.0 if latest_close >= vwap else 0.0
    else:
        vwap = 0.0
        vwap_distance = 0.0
        above_vwap = 0.0

    if len(day_data) > recent_window * 2:
        previous_window = day_data.iloc[-recent_window * 2 : -recent_window]
        previous_signed = signed_volume.loc[previous_window.index]
        previous_abs = float(previous_window[volume_column].fillna(0).sum())
        recent_signed = signed_volume.loc[recent.index]
        recent_abs = float(recent_volume.sum())
        recent_pressure = ratio(float(recent_signed.sum()), recent_abs)
        previous_pressure = ratio(float(previous_signed.sum()), previous_abs)
        volume_acceleration = recent_pressure - previous_pressure
    else:
        volume_acceleration = ratio(float(signed_volume.sum()), total_volume)

    high_value = float(day_data[high_column].max()) if high_column else float(day_data[close_column].max())
    low_value = float(day_data[low_column].min()) if low_column else float(day_data[close_column].min())
    close_position = ratio(latest_close - low_value, high_value - low_value, 0.5)
    upper_shadow_ratio = ratio(high_value - latest_close, high_value - low_value, 0.0)

    recent_volume_ratio = ratio(recent_total_volume, total_volume)
    early_high = float(day_data.head(max(1, recent_window))[close_column].max())
    breakthrough = 1.0 if latest_close >= early_high and latest_close > first_open else 0.0

    if total_volume > 0 and len(day_data) >= 3:
        pile_threshold = volume.quantile(0.80) if len(volume) >= 10 else volume.mean()
        pile_mask = volume >= pile_threshold
        pile_volume = float(volume[pile_mask].sum())
        pile_signed_volume = float(signed_volume[pile_mask].sum())
        volume_pile_ratio = ratio(pile_volume, total_volume)
        volume_pile_bias = ratio(pile_signed_volume, pile_volume)

        pile_indices = list(day_data.index[pile_mask])
        last_pile_position = day_data.index.get_loc(pile_indices[-1]) if pile_indices else -1
        if last_pile_position >= 0:
            pile_close = float(day_data.iloc[last_pile_position][close_column])
            post_window = day_data.iloc[last_pile_position + 1 : last_pile_position + 1 + max(3, recent_window // 3)]
            if post_window.empty:
                post_close = latest_close
                post_volume = pd.Series(dtype=float)
                post_signed = pd.Series(dtype=float)
            else:
                post_close = float(post_window.iloc[-1][close_column])
                post_volume = post_window[volume_column].fillna(0)
                post_signed = signed_volume.loc[post_window.index]

            price_reaction = ratio(post_close - pile_close, pile_close)
            post_pile_pressure = ratio(float(post_signed.sum()), float(post_volume.sum()))
            reaction_direction = 1.0 if volume_pile_bias >= 0 else -1.0
            volume_pile_reaction = reaction_direction * price_reaction
        else:
            post_pile_pressure = 0.0
            volume_pile_reaction = 0.0
    else:
        volume_pile_ratio = 0.0
        volume_pile_bias = 0.0
        post_pile_pressure = 0.0
        volume_pile_reaction = 0.0

    return {
        "minute_count": float(len(day_data)),
        "total_volume": total_volume,
        "up_volume": up_volume,
        "down_volume": down_volume,
        "neutral_volume": neutral_volume,
        "buy_strength": buy_strength,
        "recent_buy_strength": recent_buy_strength,
        "big_volume_buy_strength": big_volume_buy_strength,
        "pressure_score": pressure_score,
        "day_return": day_return,
        "vwap": vwap,
        "vwap_distance": vwap_distance,
        "above_vwap": above_vwap,
        "momentum_recent": momentum_recent,
        "volume_acceleration": volume_acceleration,
        "upper_shadow_ratio": upper_shadow_ratio,
        "volume_pile_ratio": volume_pile_ratio,
        "volume_pile_bias": volume_pile_bias,
        "volume_pile_reaction": volume_pile_reaction,
        "post_pile_pressure": post_pile_pressure,
        "close_position": close_position,
        "recent_volume_ratio": recent_volume_ratio,
        "breakthrough": breakthrough,
        "latest_close": latest_close,
    }


FEATURE_COLUMNS = (
    "buy_strength",
    "big_volume_buy_strength",
    "pressure_score",
    "vwap_distance",
    "close_position",
    "upper_shadow_ratio",
    "volume_pile_ratio",
    "volume_pile_bias",
    "volume_pile_reaction",
    "post_pile_pressure",
    "recent_volume_ratio",
    "breakthrough",
)


def calculate_recent_signal_score(features: dict[str, float]) -> float:
    score = 50.0
    score += (features["recent_buy_strength"] - 0.5) * 18.0
    score += features["volume_acceleration"] * 3.0
    score += features["momentum_recent"] * 60.0
    return clamp(score)


def calculate_rule_score(features: dict[str, float]) -> float:
    score = 50.0
    score += features["volume_pile_bias"] * features["volume_pile_ratio"] * 150.0
    score += features["volume_pile_reaction"] * 4200.0
    score += features["post_pile_pressure"] * 28.0
    score += (features["buy_strength"] - 0.5) * 28.0
    score += (features["big_volume_buy_strength"] - 0.5) * 20.0
    score += (features["recent_buy_strength"] - 0.5) * 10.0
    score += features["pressure_score"] * 650.0
    score += features["volume_acceleration"] * 1.5
    score += features["vwap_distance"] * 60.0
    score += (features["close_position"] - 0.5) * 4.0
    score -= features["upper_shadow_ratio"] * 6.0
    score += features["day_return"] * 10.0
    return clamp(score)


def build_training_frame(
    data: pd.DataFrame,
    columns: dict[str, str | None],
    recent_window: int,
    label_return: float,
) -> pd.DataFrame:
    rows = []
    for trade_date in get_available_dates(data):
        day_data = data[data["_trade_date"] == trade_date]
        if len(day_data) < max(20, recent_window):
            continue
        features = calculate_day_features(day_data, columns, recent_window)
        features["trade_date"] = trade_date
        features["label"] = 1 if features["day_return"] >= label_return else 0
        rows.append(features)
    return pd.DataFrame(rows)


def train_model(
    data: pd.DataFrame,
    columns: dict[str, str | None],
    stock_code: str,
    recent_window: int,
    label_return: float,
) -> None:
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("训练模型需要 scikit-learn 和 joblib，请先安装后再运行 --train-model") from exc

    frame = build_training_frame(data, columns, recent_window, label_return)
    if len(frame) < 10:
        raise ValueError(f"可训练样本太少，目前只有 {len(frame)} 天，建议至少 10 天以上")
    if frame["label"].nunique() < 2:
        raise ValueError("训练标签只有单一类别，请调低或调高 --label-return")

    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.3,
        random_state=42,
        stratify=frame["label"],
    )
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(train_frame[list(FEATURE_COLUMNS)], train_frame["label"])
    prediction = pipeline.predict(test_frame[list(FEATURE_COLUMNS)])
    accuracy = accuracy_score(test_frame["label"], prediction)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(stock_code)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "recent_window": recent_window,
            "label_return": label_return,
            "sample_count": len(frame),
        },
        path,
    )
    print(f"模型已保存: {path}")
    print(f"训练样本: {len(train_frame)} 天，测试样本: {len(test_frame)} 天，测试准确率: {accuracy:.2%}")
    print(f"标签定义: 当日收盘相对开盘涨幅 >= {label_return:.2%}")


def predict_model_score(stock_code: str, features: dict[str, float]) -> float | None:
    stock_path = model_path(stock_code)
    path = MARKET_MODEL_PATH if MARKET_MODEL_PATH.exists() else stock_path
    if not path.exists():
        return None

    try:
        import joblib
    except ImportError:
        return None

    payload = joblib.load(path)
    pipeline = payload["pipeline"]
    feature_columns = payload["feature_columns"]
    feature_frame = pd.DataFrame([{name: features.get(name, 0.0) for name in feature_columns}])
    probability = float(pipeline.predict_proba(feature_frame)[0][1])
    return probability * 100.0


def calculate_strength(
    data: pd.DataFrame,
    stock_code: str,
    trade_date: str,
    columns: dict[str, str | None],
    recent_window: int = 30,
    use_model: bool = False,
) -> StrengthResult:
    day_data = data[data["_trade_date"] == trade_date].copy()
    if day_data.empty:
        available_dates = ", ".join(get_available_dates(data))
        raise ValueError(f"{trade_date} 没有分时数据。可用日期: {available_dates}")

    features = calculate_day_features(day_data, columns, recent_window)
    rule_score = calculate_rule_score(features)
    recent_signal_score = calculate_recent_signal_score(features)
    model_score = None
    final_score = rule_score

    vwap = features["vwap"] if features["total_volume"] > 0 else None
    above_vwap = bool(features["above_vwap"]) if features["total_volume"] > 0 else None

    return StrengthResult(
        stock_code=stock_code,
        trade_date=trade_date,
        minute_count=int(features["minute_count"]),
        total_volume=features["total_volume"],
        up_volume=features["up_volume"],
        down_volume=features["down_volume"],
        neutral_volume=features["neutral_volume"],
        buy_strength=features["buy_strength"],
        recent_buy_strength=features["recent_buy_strength"],
        big_volume_buy_strength=features["big_volume_buy_strength"],
        pressure_score=features["pressure_score"],
        vwap_distance=features["vwap_distance"],
        momentum_recent=features["momentum_recent"],
        volume_acceleration=features["volume_acceleration"],
        upper_shadow_ratio=features["upper_shadow_ratio"],
        volume_pile_ratio=features["volume_pile_ratio"],
        volume_pile_bias=features["volume_pile_bias"],
        volume_pile_reaction=features["volume_pile_reaction"],
        post_pile_pressure=features["post_pile_pressure"],
        close_position=features["close_position"],
        recent_signal_score=recent_signal_score,
        rule_score=rule_score,
        model_score=model_score,
        final_score=final_score,
        latest_close=features["latest_close"],
        vwap=vwap,
        above_vwap=above_vwap,
    )


def calculate_strength_for_stock(
    stock_code: str,
    trade_date: str | None = None,
    recent_window: int = 30,
    minute_data: pd.DataFrame | None = None,
) -> StrengthResult:
    raw_df = minute_data
    if raw_df is None:
        raw_df, _ = get_minute_em(stock_code, trade_date)

    data, columns = prepare_minute_data(raw_df)
    target_date = normalize_trade_date(trade_date) if trade_date else get_available_dates(data)[-1]
    return calculate_strength(
        data=data,
        stock_code=stock_code,
        trade_date=target_date,
        columns=columns,
        recent_window=recent_window,
        use_model=False,
    )


def describe_strength(score: float) -> str:
    if score >= 75:
        return "买盘很强"
    if score >= 65:
        return "买盘偏强"
    if score >= 50:
        return "买盘略占优"
    if score >= 40:
        return "多空偏弱"
    return "卖盘强"


def print_result(result: StrengthResult) -> None:
    vwap_text = "--" if result.vwap is None else f"{result.vwap:.3f}"
    above_vwap_text = "--" if result.above_vwap is None else ("是" if result.above_vwap else "否")
    model_text = "--" if result.model_score is None else f"{result.model_score:.2f}/100"

    print(f"股票代码: {result.stock_code}")
    print(f"交易日期: {result.trade_date}")
    print(f"分钟数量: {result.minute_count}")
    print(f"总成交量: {result.total_volume:.0f}")
    print(f"上涨分钟成交量: {result.up_volume:.0f}")
    print(f"下跌分钟成交量: {result.down_volume:.0f}")
    print(f"平衡分钟成交量: {result.neutral_volume:.0f}")
    print(f"全天买盘占比: {result.buy_strength * 100:.2f}%")
    print(f"最近窗口买盘占比: {result.recent_buy_strength * 100:.2f}%")
    print(f"大成交量分钟买盘占比: {result.big_volume_buy_strength * 100:.2f}%")
    print(f"价格压力值: {result.pressure_score:.6f}")
    print(f"VWAP偏离: {result.vwap_distance * 100:.2f}%")
    print(f"最近动量: {result.momentum_recent * 100:.2f}%")
    print(f"量能加速度: {result.volume_acceleration:.4f}")
    print(f"上影线比例: {result.upper_shadow_ratio * 100:.2f}%")
    print(f"量堆成交占比: {result.volume_pile_ratio * 100:.2f}%")
    print(f"量堆买卖方向: {result.volume_pile_bias:.4f}")
    print(f"量堆后价格反应: {result.volume_pile_reaction * 100:.2f}%")
    print(f"量堆后买卖压力: {result.post_pile_pressure:.4f}")
    print(f"收盘位置: {result.close_position * 100:.2f}%")
    print(f"短线异动强度: {result.recent_signal_score:.2f}/100")
    print(f"全日规则强度: {result.rule_score:.2f}/100")
    print(f"模型强度: {model_text}")
    print(f"综合分时成交强度: {result.final_score:.2f}/100 ({describe_strength(result.final_score)})")
    print(f"最新价: {result.latest_close:.3f}")
    print(f"VWAP: {vwap_text}")
    print(f"最新价高于VWAP: {above_vwap_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算指定日期的股票分时成交强度")
    parser.add_argument("-s", "--stock", help="股票代码，例如 002119 或 sh600000")
    parser.add_argument("-d", "--date", help="交易日期，格式 YYYY-MM-DD 或 YYYYMMDD")
    parser.add_argument("--list-dates", action="store_true", help="只显示拉取数据里包含的日期")
    parser.add_argument("--recent-window", type=int, default=30, help="最近窗口分钟数，默认 30")
    parser.add_argument("--train-model", action="store_true", help="用本次拉取到的多日分时数据训练模型")
    parser.add_argument("--use-model", action="store_true", help="加载已训练模型并融合到综合评分")
    parser.add_argument("--label-return", type=float, default=0.005, help="训练标签阈值，默认 0.005 表示 0.5%%")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stock_code = normalize_stock_code(args.stock or input("请输入股票代码，例如 002119 或 sh600000: "))
    if not stock_code:
        raise ValueError("股票代码不能为空")

    raw_df, _ = get_minute_em(stock_code, args.date)
    data, columns = prepare_minute_data(raw_df)

    available_dates = get_available_dates(data)
    if args.list_dates:
        print("可用日期:")
        for date in available_dates:
            print(date)
        return

    if args.train_model:
        train_model(data, columns, stock_code, args.recent_window, args.label_return)
        return

    trade_date = normalize_trade_date(args.date or input("请输入交易日期(YYYY-MM-DD 或 YYYYMMDD): "))
    result = calculate_strength(
        data=data,
        stock_code=stock_code,
        trade_date=trade_date,
        columns=columns,
        recent_window=args.recent_window,
        use_model=args.use_model,
    )
    print_result(result)


if __name__ == "__main__":
    main()
