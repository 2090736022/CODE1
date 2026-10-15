import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import ruptures as rpt
except ImportError:
    rpt = None

from model import (
    END_DATE,
    MODEL_DIR,
    START_DATE,
    create_pro,
    ensure_dirs,
    load_or_download_panel,
    load_stock_universe,
)
from progress import show_progress


# =====================
# model2: trend strength + turning point model
# =====================

MODEL2_BUNDLE_FILE = MODEL_DIR / "trend_turn_bundle.pkl"
MODEL2_METRICS_FILE = MODEL_DIR / "trend_turn_metrics.json"
MODEL2_LATEST_PRED_FILE = MODEL_DIR / "trend_turn_latest_predictions.csv"

TRAIN_END_DATE = "20221231"
VALID_END_DATE = "20241231"
TEST_END_DATE = "20251231"
SPLIT_GAP_DAYS = 15

RANDOM_STATE = 42
MIN_STOCK_ROWS = 180
SLOPE_TARGET_DAYS = 20
TURN_LOOKAHEAD_DAYS = 10
PELT_MIN_SIZE = 20
PELT_JUMP = 5
PELT_PENALTY = 4.0
HMM_STATES = 3
DBSCAN_EPS = 0.85
DBSCAN_MIN_SAMPLES = 8
GARCH_LOOKBACK_ROWS = 260
MAX_TRAIN_ROWS = 500_000
MAX_VALID_ROWS = 250_000
MAX_TEST_ROWS = 250_000
MAX_HMM_TRAIN_ROWS = 200_000
MAX_DBSCAN_FIT_ROWS = 20_000
PREDICT_BATCH_ROWS = 200_000

ROLLING_WINDOWS = (5, 10, 20, 60, 120)
FEATURE_COLUMNS = [
    "ret_1d",
    "range_pct",
    "body_pct",
    "upper_shadow",
    "lower_shadow",
    "volume_chg",
    "turnover_rate",
    "log_mv",
    "slope_5",
    "slope_10",
    "slope_20",
    "slope_60",
    "slope_120",
    "ret_mean_5",
    "ret_mean_10",
    "ret_mean_20",
    "ret_mean_60",
    "ret_mean_120",
    "ret_std_5",
    "ret_std_10",
    "ret_std_20",
    "ret_std_60",
    "ret_std_120",
    "ma_gap_5",
    "ma_gap_10",
    "ma_gap_20",
    "ma_gap_60",
    "ma_gap_120",
    "volatility_z",
    "garch_vol",
    "garch_exhaustion",
    "hmm_bull_prob",
    "hmm_bear_prob",
    "hmm_chop_prob",
]


def _safe_ratio(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _linear_slope(values):
    y = np.asarray(values, dtype="float64")
    if len(y) < 2 or np.isnan(y).any():
        return np.nan
    x = np.arange(len(y), dtype="float64")
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return np.nan
    return float(np.sum(x * (y - y.mean())) / denom)


def _future_slope(close, horizon=SLOPE_TARGET_DAYS):
    log_close = np.log(close.replace(0, np.nan))
    return log_close.rolling(horizon).apply(_linear_slope, raw=False).shift(-horizon)


def _fallback_pelt_l2(series, penalty=PELT_PENALTY, min_size=PELT_MIN_SIZE):
    y = np.asarray(series, dtype="float64")
    y = y[np.isfinite(y)]
    n = len(y)
    if n < min_size * 2:
        return []

    prefix = np.concatenate([[0.0], np.cumsum(y)])
    prefix2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def segment_cost(start, end):
        length = end - start
        if length <= 0:
            return 0.0
        total = prefix[end] - prefix[start]
        total2 = prefix2[end] - prefix2[start]
        return total2 - total * total / length

    costs = np.full(n + 1, np.inf)
    paths = [[] for _ in range(n + 1)]
    costs[0] = -penalty
    candidates = [0]

    for end in range(min_size, n + 1):
        valid = [start for start in candidates if end - start >= min_size]
        if not valid:
            candidates.append(end - min_size + 1)
            continue

        scored = [(costs[start] + segment_cost(start, end) + penalty, start) for start in valid]
        best_cost, best_start = min(scored, key=lambda item: item[0])
        costs[end] = best_cost
        paths[end] = paths[best_start] + [best_start]
        candidates = [start for cost, start in scored if cost <= best_cost + penalty]
        candidates.append(end - min_size + 1)

    return [point for point in paths[n] if point > 0]


def detect_pelt_change_points(log_close):
    values = log_close.dropna().to_numpy(dtype="float64")
    if len(values) < MIN_STOCK_ROWS:
        return []

    if rpt is not None:
        algo = rpt.Pelt(model="l2", min_size=PELT_MIN_SIZE, jump=PELT_JUMP).fit(values)
        points = algo.predict(pen=PELT_PENALTY)
        return [point for point in points if point < len(values)]

    return _fallback_pelt_l2(values)


def mark_bullish_turn_labels(group):
    group = group.sort_values("trade_date").copy()
    log_close = np.log(group["close"].replace(0, np.nan))
    change_points = detect_pelt_change_points(log_close)
    labels = np.zeros(len(group), dtype="int8")

    for cp in change_points:
        if cp < 20 or cp + TURN_LOOKAHEAD_DAYS >= len(group):
            continue
        prev_slope = _linear_slope(log_close.iloc[cp - 20 : cp])
        next_slope = _linear_slope(log_close.iloc[cp : cp + TURN_LOOKAHEAD_DAYS])
        if pd.notna(prev_slope) and pd.notna(next_slope) and prev_slope <= 0 < next_slope:
            start = max(0, cp - TURN_LOOKAHEAD_DAYS)
            labels[start : cp + 1] = 1

    group["bullish_turn_label"] = labels
    return group


def add_garch_features(group):
    group = group.sort_values("trade_date").copy()
    returns = (100 * group["ret_1d"]).replace([np.inf, -np.inf], np.nan).dropna()

    group["garch_vol"] = group["ret_std_20"]
    if len(returns) >= GARCH_LOOKBACK_ROWS:
        try:
            fit_returns = returns.tail(GARCH_LOOKBACK_ROWS)
            model = arch_model(fit_returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
            result = model.fit(disp="off")
            vol = pd.Series(result.conditional_volatility / 100.0, index=fit_returns.index)
            group.loc[vol.index, "garch_vol"] = vol.astype("float32")
        except Exception:
            pass

    vol_ma = group["garch_vol"].rolling(60, min_periods=20).mean()
    vol_std = group["garch_vol"].rolling(60, min_periods=20).std()
    group["garch_exhaustion"] = ((group["garch_vol"] - vol_ma) / vol_std).clip(lower=0)
    group["garch_exhaustion"] = group["garch_exhaustion"].fillna(0)
    return group


def prepare_base_features(panel):
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "vol",
        "amount",
        "turnover_rate",
        "total_mv",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")

    grouped = data.groupby("ts_code", group_keys=False)
    data["ret_1d"] = _safe_ratio(data["close"], data["pre_close"]) - 1
    data["range_pct"] = _safe_ratio(data["high"] - data["low"], data["pre_close"])
    data["body_pct"] = _safe_ratio(data["close"] - data["open"], data["pre_close"])
    data["upper_shadow"] = _safe_ratio(data["high"] - data[["open", "close"]].max(axis=1), data["pre_close"])
    data["lower_shadow"] = _safe_ratio(data[["open", "close"]].min(axis=1) - data["low"], data["pre_close"])
    data["volume_chg"] = _safe_ratio(data["vol"], grouped["vol"].shift(1)) - 1
    data["log_mv"] = np.log1p(data["total_mv"])
    data["log_close"] = np.log(data["close"].replace(0, np.nan))

    show_progress(0, len(ROLLING_WINDOWS), "model2 rolling features")
    for window in ROLLING_WINDOWS:
        data[f"slope_{window}"] = grouped["log_close"].transform(
            lambda s, w=window: s.rolling(w).apply(_linear_slope, raw=False)
        )
        data[f"ret_mean_{window}"] = grouped["ret_1d"].transform(lambda s, w=window: s.rolling(w).mean())
        data[f"ret_std_{window}"] = grouped["ret_1d"].transform(lambda s, w=window: s.rolling(w).std())
        close_ma = grouped["close"].transform(lambda s, w=window: s.rolling(w).mean())
        data[f"ma_gap_{window}"] = _safe_ratio(data["close"], close_ma) - 1
        show_progress(ROLLING_WINDOWS.index(window) + 1, len(ROLLING_WINDOWS), "model2 rolling features")

    vol_mean = grouped["ret_std_20"].transform(lambda s: s.rolling(120, min_periods=30).mean())
    vol_std = grouped["ret_std_20"].transform(lambda s: s.rolling(120, min_periods=30).std())
    data["volatility_z"] = (data["ret_std_20"] - vol_mean) / vol_std
    data["future_slope"] = grouped["close"].transform(_future_slope)

    total_groups = data["ts_code"].nunique()
    processed_frames = []
    show_progress(0, total_groups, "model2 GARCH/PELT")
    for index, (_, group) in enumerate(data.groupby("ts_code", group_keys=False), start=1):
        group = add_garch_features(group)
        processed_frames.append(mark_bullish_turn_labels(group))
        show_progress(index, total_groups, "model2 GARCH/PELT")
    data = pd.concat(processed_frames, ignore_index=True)
    del processed_frames
    return data.replace([np.inf, -np.inf], np.nan)


def add_hmm_probabilities(data):
    hmm_columns = ["ret_1d", "range_pct", "volume_chg"]
    train_mask = (data["trade_date"] <= pd.Timestamp(TRAIN_END_DATE)) & data[hmm_columns].notna().all(axis=1)
    x_train = data.loc[train_mask, hmm_columns].copy()
    if len(x_train) > MAX_HMM_TRAIN_ROWS:
        x_train = x_train.sample(MAX_HMM_TRAIN_ROWS, random_state=RANDOM_STATE).sort_index()
    if len(x_train) < 1000:
        raise ValueError("HMM 训练样本太少，请检查行情数据")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train).astype("float32")
    show_progress(0, 3, "model2 HMM states")
    hmm = GaussianHMM(
        n_components=HMM_STATES,
        covariance_type="full",
        n_iter=300,
        random_state=RANDOM_STATE,
    )
    hmm.fit(x_scaled)
    show_progress(1, 3, "model2 HMM states")

    all_mask = data[hmm_columns].notna().all(axis=1)
    all_indices = data.index[all_mask].to_numpy()
    show_progress(2, 3, "model2 HMM states")
    state_returns = []
    train_probs = hmm.predict_proba(x_scaled)
    for state in range(HMM_STATES):
        weights = train_probs[:, state]
        state_returns.append(float(np.average(x_train["ret_1d"], weights=weights)))

    bull_state = int(np.argmax(state_returns))
    bear_state = int(np.argmin(state_returns))
    chop_state = next(state for state in range(HMM_STATES) if state not in {bull_state, bear_state})

    data["hmm_bull_prob"] = np.nan
    data["hmm_bear_prob"] = np.nan
    data["hmm_chop_prob"] = np.nan

    for start in range(0, len(all_indices), PREDICT_BATCH_ROWS):
        batch_indices = all_indices[start : start + PREDICT_BATCH_ROWS]
        batch_x = scaler.transform(data.loc[batch_indices, hmm_columns]).astype("float32")
        batch_probs = hmm.predict_proba(batch_x)
        data.loc[batch_indices, "hmm_bull_prob"] = batch_probs[:, bull_state].astype("float32")
        data.loc[batch_indices, "hmm_bear_prob"] = batch_probs[:, bear_state].astype("float32")
        data.loc[batch_indices, "hmm_chop_prob"] = batch_probs[:, chop_state].astype("float32")
    show_progress(3, 3, "model2 HMM states")

    hmm_meta = {
        "columns": hmm_columns,
        "bull_state": bull_state,
        "bear_state": bear_state,
        "chop_state": chop_state,
        "state_returns": state_returns,
    }
    return data, scaler, hmm, hmm_meta


def add_dbscan_noise_flag(data):
    cluster_columns = ["slope_20", "slope_60", "ret_std_20", "ma_gap_20", "hmm_chop_prob"]
    mask = data[cluster_columns].notna().all(axis=1)
    data["dbscan_noise"] = 1
    data["dbscan_label"] = -1
    if mask.sum() < DBSCAN_MIN_SAMPLES * 5:
        return data, None, None

    show_progress(0, 3, "model2 DBSCAN noise")
    fit_frame = data.loc[mask, cluster_columns]
    if len(fit_frame) > MAX_DBSCAN_FIT_ROWS:
        fit_frame = fit_frame.sample(MAX_DBSCAN_FIT_ROWS, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit_frame).astype("float32")
    show_progress(1, 3, "model2 DBSCAN noise")
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    fit_labels = dbscan.fit_predict(x_fit)
    show_progress(2, 3, "model2 DBSCAN noise")

    cluster_frame = fit_frame.copy()
    cluster_frame["label"] = fit_labels
    choppy_labels = set(cluster_frame.loc[cluster_frame["label"] == -1, "label"].unique().tolist())
    slope_abs_median = fit_frame["slope_20"].abs().median()
    chop_median = fit_frame["hmm_chop_prob"].median()
    for label, group in cluster_frame.groupby("label"):
        if label == -1:
            continue
        low_slope = group["slope_20"].abs().median() < slope_abs_median
        high_chop = group["hmm_chop_prob"].median() > chop_median
        if low_slope and high_chop:
            choppy_labels.add(label)

    valid_core = fit_labels != -1
    if valid_core.any():
        neighbor = NearestNeighbors(n_neighbors=1, algorithm="auto")
        neighbor.fit(x_fit[valid_core])
        core_labels = fit_labels[valid_core]
        all_indices = data.index[mask].to_numpy()

        for start in range(0, len(all_indices), PREDICT_BATCH_ROWS):
            batch_indices = all_indices[start : start + PREDICT_BATCH_ROWS]
            batch_x = scaler.transform(data.loc[batch_indices, cluster_columns]).astype("float32")
            distances, nearest = neighbor.kneighbors(batch_x, return_distance=True)
            batch_labels = core_labels[nearest[:, 0]]
            batch_labels = np.where(distances[:, 0] <= DBSCAN_EPS, batch_labels, -1)
            data.loc[batch_indices, "dbscan_label"] = batch_labels.astype("int32")
            data.loc[batch_indices, "dbscan_noise"] = np.isin(batch_labels, list(choppy_labels)).astype("int8")

    show_progress(3, 3, "model2 DBSCAN noise")
    return data, scaler, {
        "columns": cluster_columns,
        "choppy_labels": sorted(map(int, choppy_labels)),
        "fit_rows": int(len(fit_frame)),
        "max_fit_rows": int(MAX_DBSCAN_FIT_ROWS),
    }


def build_model2_frame(panel):
    data = prepare_base_features(panel)
    data, hmm_scaler, hmm_model, hmm_meta = add_hmm_probabilities(data)
    data, dbscan_scaler, dbscan_meta = add_dbscan_noise_flag(data)

    keep_columns = [
        "ts_code",
        "trade_date",
        "future_slope",
        "bullish_turn_label",
        "dbscan_noise",
        "dbscan_label",
    ] + FEATURE_COLUMNS
    frame = data[keep_columns].dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    for column in FEATURE_COLUMNS + ["future_slope"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    frame["bullish_turn_label"] = frame["bullish_turn_label"].astype("int8")
    frame["dbscan_noise"] = frame["dbscan_noise"].astype("int8")
    frame["dbscan_label"] = frame["dbscan_label"].astype("int32")

    meta = {
        "hmm_scaler": hmm_scaler,
        "hmm_model": hmm_model,
        "hmm_meta": hmm_meta,
        "dbscan_scaler": dbscan_scaler,
        "dbscan_meta": dbscan_meta,
    }
    return frame, meta


def split_frames(frame):
    gap = pd.Timedelta(days=SPLIT_GAP_DAYS)
    train_end = pd.Timestamp(TRAIN_END_DATE) - gap
    valid_start = pd.Timestamp(TRAIN_END_DATE) + gap
    valid_end = pd.Timestamp(VALID_END_DATE) - gap
    test_start = pd.Timestamp(VALID_END_DATE) + gap
    test_end = pd.Timestamp(TEST_END_DATE)

    train = frame[(frame["trade_date"] <= train_end) & (frame["dbscan_noise"] == 0)].dropna(
        subset=["future_slope", "bullish_turn_label"]
    )
    valid = frame[
        (frame["trade_date"] >= valid_start)
        & (frame["trade_date"] <= valid_end)
        & (frame["dbscan_noise"] == 0)
    ].dropna(subset=["future_slope", "bullish_turn_label"])
    test = frame[
        (frame["trade_date"] >= test_start)
        & (frame["trade_date"] <= test_end)
        & (frame["dbscan_noise"] == 0)
    ].dropna(subset=["future_slope", "bullish_turn_label"])

    if train.empty or valid.empty or test.empty:
        raise ValueError("model2 时间切分后存在空数据集，请检查日期范围或过滤强度")

    if len(train) > MAX_TRAIN_ROWS:
        train = train.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE).sort_values("trade_date")
    if len(valid) > MAX_VALID_ROWS:
        valid = valid.sample(MAX_VALID_ROWS, random_state=RANDOM_STATE).sort_values("trade_date")
    if len(test) > MAX_TEST_ROWS:
        test = test.sample(MAX_TEST_ROWS, random_state=RANDOM_STATE).sort_values("trade_date")

    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def make_regressor():
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=120,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_classifier():
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=120,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_bundle(train, valid, meta):
    x_train = train[FEATURE_COLUMNS].astype("float32", copy=False)
    x_valid = valid[FEATURE_COLUMNS].astype("float32", copy=False)

    show_progress(0, 2, "model2 train models")
    regressor = make_regressor()
    regressor.fit(
        x_train,
        train["future_slope"],
        eval_set=[(x_valid, valid["future_slope"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(200)],
    )
    show_progress(1, 2, "model2 train models")

    classifier = make_classifier()
    classifier.fit(
        x_train,
        train["bullish_turn_label"],
        eval_set=[(x_valid, valid["bullish_turn_label"])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(200)],
    )
    show_progress(2, 2, "model2 train models")

    return {
        "feature_columns": FEATURE_COLUMNS,
        "regressor": regressor,
        "classifier": classifier,
        "meta": meta,
        "params": {
            "slope_target_days": SLOPE_TARGET_DAYS,
            "turn_lookahead_days": TURN_LOOKAHEAD_DAYS,
            "pelt_penalty": PELT_PENALTY,
            "hmm_states": HMM_STATES,
            "dbscan_eps": DBSCAN_EPS,
        },
    }


def predict_with_bundle(bundle, frame):
    pred = frame[["ts_code", "trade_date", "dbscan_noise", "garch_exhaustion", "hmm_bull_prob", "hmm_chop_prob"]].copy()
    pred_slope = np.empty(len(frame), dtype="float32")
    turn_prob = np.empty(len(frame), dtype="float32")

    for start in range(0, len(frame), PREDICT_BATCH_ROWS):
        end = min(start + PREDICT_BATCH_ROWS, len(frame))
        x = frame.iloc[start:end][bundle["feature_columns"]].astype("float32", copy=False)
        pred_slope[start:end] = bundle["regressor"].predict(
            x,
            num_iteration=bundle["regressor"].best_iteration_,
        ).astype("float32")
        turn_prob[start:end] = bundle["classifier"].predict_proba(
            x,
            num_iteration=bundle["classifier"].best_iteration_,
        )[:, 1].astype("float32")

    pred["pred_slope"] = pred_slope
    pred["turn_prob"] = turn_prob
    pred["trend_strength"] = pred["pred_slope"] * (1.0 + pred["hmm_bull_prob"] - pred["hmm_chop_prob"])
    pred["score"] = pred["trend_strength"] * pred["turn_prob"] / (1.0 + pred["garch_exhaustion"].clip(lower=0))
    return pred.sort_values("score", ascending=False).reset_index(drop=True)


def evaluate_bundle(bundle, test):
    pred = predict_with_bundle(bundle, test)
    metrics = {
        "future_slope": {
            "mae": float(mean_absolute_error(test["future_slope"], pred["pred_slope"])),
            "actual_mean": float(test["future_slope"].mean()),
            "pred_mean": float(pred["pred_slope"].mean()),
        },
        "bullish_turn": {
            "positive_rate": float(test["bullish_turn_label"].mean()),
            "mean_prob": float(pred["turn_prob"].mean()),
            "auc": None,
        },
        "filters": {
            "test_rows": int(len(test)),
            "dbscan_noise_rate_all_frame": None,
        },
    }
    if test["bullish_turn_label"].nunique() > 1:
        metrics["bullish_turn"]["auc"] = float(roc_auc_score(test["bullish_turn_label"], pred["turn_prob"]))
    return metrics, pred


def save_bundle(bundle, metrics):
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL2_BUNDLE_FILE, "wb") as file:
        pickle.dump(bundle, file)
    with open(MODEL2_METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def build_latest_predictions(bundle, frame):
    show_progress(0, 3, "model2 latest predictions")
    latest = (
        frame.sort_values(["ts_code", "trade_date"])
        .groupby("ts_code", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    show_progress(1, 3, "model2 latest predictions")
    latest = latest[(latest["dbscan_noise"] == 0) & (latest["garch_exhaustion"] <= 2.0)].reset_index(drop=True)
    pred = predict_with_bundle(bundle, latest)
    show_progress(2, 3, "model2 latest predictions")
    pred.to_csv(MODEL2_LATEST_PRED_FILE, index=False, encoding="utf-8-sig")
    show_progress(3, 3, "model2 latest predictions")
    return pred


def print_metrics(metrics):
    slope = metrics["future_slope"]
    turn = metrics["bullish_turn"]
    auc_text = "NA" if turn["auc"] is None else f"{turn['auc']:.4f}"
    print("\n[model2 metrics]")
    print(f"future_slope: MAE={slope['mae']:.6f}, actual_mean={slope['actual_mean']:.6f}, pred_mean={slope['pred_mean']:.6f}")
    print(f"bullish_turn: AUC={auc_text}, mean_prob={turn['mean_prob']:.4f}, positive_rate={turn['positive_rate']:.4f}")


def main():
    ensure_dirs()
    print("[model2] 1/6 load stock universe")
    stock_codes = load_stock_universe()
    print(f"[model2] universe: {len(stock_codes)}")

    print("[model2] 2/6 load/download panel")
    pro = create_pro()
    panel = load_or_download_panel(pro, stock_codes, START_DATE, END_DATE)
    print(f"[model2] panel rows: {len(panel)}")

    print("[model2] 3/6 build PELT/HMM/GARCH/DBSCAN features")
    frame, meta = build_model2_frame(panel)
    print(f"[model2] feature rows: {len(frame)}, noise_rate={frame['dbscan_noise'].mean():.2%}")

    print("[model2] 4/6 split and train")
    train, valid, test = split_frames(frame)
    print(f"[model2] train={len(train)}, valid={len(valid)}, test={len(test)}")
    bundle = train_bundle(train, valid, meta)

    print("[model2] 5/6 evaluate")
    metrics, _ = evaluate_bundle(bundle, test)
    metrics["filters"]["dbscan_noise_rate_all_frame"] = float(frame["dbscan_noise"].mean())

    print("[model2] 6/6 save bundle and latest predictions")
    latest = build_latest_predictions(bundle, frame)
    save_bundle(bundle, metrics)

    print_metrics(metrics)
    print(f"[model2] saved bundle: {MODEL2_BUNDLE_FILE}")
    print(f"[model2] saved metrics: {MODEL2_METRICS_FILE}")
    print(f"[model2] saved latest predictions: {MODEL2_LATEST_PRED_FILE}")
    print("\n[model2 top 10 latest]")
    print(latest.head(10))


if __name__ == "__main__":
    main()
