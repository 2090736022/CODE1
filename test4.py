from datetime import datetime, timedelta
from multiprocessing import freeze_support
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tushare as ts
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

LOCAL_OPENFE_DIR = Path(__file__).resolve().parent / "OpenFE"
if LOCAL_OPENFE_DIR.exists():
    sys.path.insert(0, str(LOCAL_OPENFE_DIR))

import openfe.openfe as openfe_module
from openfe import OpenFE, transform, tree_to_formula

import config


TS_CODE = "000001.SZ"
ASSET = "E"
YEARS = 3
PRED_DAYS = 5
TOP_FEATURES = 20
N_JOBS = 4
RANDOM_STATE = 42
OUTPUT_CSV = "openfe_stock_features.csv"
OUTPUT_FEATHER = "openfe_stock_features.feather"


def _patch_openfe_mean_squared_error():
    original_mean_squared_error = mean_squared_error

    def compatible_mean_squared_error(
        y_true,
        y_pred,
        *,
        sample_weight=None,
        multioutput="uniform_average",
        squared=True,
    ):
        mse = original_mean_squared_error(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )
        return mse if squared else mse**0.5

    openfe_module.mean_squared_error = compatible_mean_squared_error


def get_start_date(years=YEARS):
    return (datetime.today() - timedelta(days=365 * years)).strftime("%Y%m%d")


def load_stock_daily(ts_code=TS_CODE, start_date=None, end_date=None):
    ts.set_token(config.TUSHARE_TOKEN)
    start_date = start_date or get_start_date()
    end_date = end_date or datetime.today().strftime("%Y%m%d")

    df = ts.pro_bar(
        ts_code=ts_code,
        asset=ASSET,
        start_date=start_date,
        end_date=end_date,
        adj="qfq",
    )
    if df is None or df.empty:
        raise ValueError(f"No Tushare data returned for {ts_code}.")

    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def build_openfe_dataset(df, pred_days=PRED_DAYS):
    data = df.copy()
    data["return_1d"] = data["close"].pct_change()
    data["return_3d"] = data["close"].pct_change(3)
    data["return_5d"] = data["close"].pct_change(5)
    data["return_10d"] = data["close"].pct_change(10)
    data["high_low_range"] = (data["high"] - data["low"]) / data["close"]
    data["open_close_gap"] = (data["close"] - data["open"]) / data["open"]
    data["volume_change"] = data["vol"].pct_change()
    data["amount_change"] = data["amount"].pct_change()

    for window in (5, 10, 20, 60):
        ma = data["close"].rolling(window).mean()
        vol_ma = data["vol"].rolling(window).mean()
        data[f"ma{window}_gap"] = data["close"] / ma - 1
        data[f"vol_ma{window}_gap"] = data["vol"] / vol_ma - 1

    data[f"future_return_{pred_days}d"] = data["close"].shift(-pred_days) / data["close"] - 1
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    label_column = f"future_return_{pred_days}d"
    drop_columns = ["ts_code", "trade_date", label_column]
    feature_data = data.drop(columns=[col for col in drop_columns if col in data.columns])
    feature_data = feature_data.select_dtypes(include=[np.number]).astype("float32")
    label = data[[label_column]].astype("float32")
    meta = data[["trade_date"]].copy()
    if "ts_code" in data.columns:
        meta.insert(0, "ts_code", data["ts_code"])

    return feature_data, label, meta


def split_by_time(data, label, meta, test_size=0.2):
    train_x, test_x, train_y, test_y, train_meta, test_meta = train_test_split(
        data,
        label,
        meta,
        test_size=test_size,
        shuffle=False,
    )
    return train_x, test_x, train_y, test_y, train_meta, test_meta


def save_features(train_x, test_x, train_meta, test_meta):
    output = pd.concat(
        [
            pd.concat([train_meta.reset_index(drop=True), train_x.reset_index(drop=True)], axis=1),
            pd.concat([test_meta.reset_index(drop=True), test_x.reset_index(drop=True)], axis=1),
        ],
        ignore_index=True,
    )
    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    try:
        output.to_feather(OUTPUT_FEATHER)
    except ImportError:
        print("pyarrow is not installed; skipped feather output.")

    return output


def main():
    _patch_openfe_mean_squared_error()

    print(f"Loading {TS_CODE} daily data from Tushare...")
    raw = load_stock_daily()
    data, label, meta = build_openfe_dataset(raw)
    if len(data) < 200:
        raise ValueError(f"Only {len(data)} usable rows; OpenFE needs more stock history.")

    train_x, test_x, train_y, test_y, train_meta, test_meta = split_by_time(data, label, meta)
    print(f"Train rows: {len(train_x)}, test rows: {len(test_x)}, base features: {train_x.shape[1]}")

    ofe = OpenFE()
    ofe.fit(
        data=train_x,
        label=train_y,
        task="regression",
        metric="rmse",
        n_jobs=N_JOBS,
        seed=RANDOM_STATE,
    )

    selected_features = ofe.new_features_list[:TOP_FEATURES]
    train_fe, test_fe = transform(train_x, test_x, selected_features, n_jobs=N_JOBS)
    output = save_features(train_fe, test_fe, train_meta, test_meta)

    print(f"Selected OpenFE features: {len(selected_features)}")
    for i, feature in enumerate(selected_features, start=1):
        print(f"{i:02d}. {tree_to_formula(feature)}")

    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved rows: {len(output)}, columns: {output.shape[1]}")


if __name__ == "__main__":
    freeze_support()
    main()
