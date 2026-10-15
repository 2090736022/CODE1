from datetime import datetime, timedelta
import math
import sys

import numpy as np
import pandas as pd
import tushare as ts

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("缺少依赖: torch")
    print("请先安装: python -m pip install torch")
    sys.exit(1)

import config


TS_CODE = "000001.SH"
YEARS = 10
SEQ_LEN = 30
PRED_DAYS = 3
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 1e-3
MODEL_FILE = "transformer_000001_sh.pt"


def get_start_date(years=YEARS):
    return (datetime.today() - timedelta(days=365 * years)).strftime("%Y%m%d")


def get_index_data(ts_code=TS_CODE, start_date=None):
    ts.set_token(config.TUSHARE_TOKEN)
    start_date = start_date or get_start_date()
    df = ts.pro_bar(ts_code=ts_code, asset="I", start_date=start_date)
    if df is None or df.empty:
        raise ValueError(f"没有拉取到 {ts_code} 指数数据")

    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def build_features(df):
    data = df.copy()
    data["return"] = data["close"].pct_change()
    data["high_low"] = (data["high"] - data["low"]) / data["close"]
    data["open_close"] = (data["close"] - data["open"]) / data["open"]
    data["volume_change"] = data["vol"].pct_change()
    data["ma5_gap"] = data["close"] / data["close"].rolling(5).mean() - 1
    data["ma20_gap"] = data["close"] / data["close"].rolling(20).mean() - 1
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return data


def standardize_features(feature_values):
    mean = feature_values.mean(axis=0)
    std = feature_values.std(axis=0)
    std[std == 0] = 1
    return (feature_values - mean) / std, mean, std


def make_dataset(data, seq_len=SEQ_LEN, pred_days=PRED_DAYS):
    feature_columns = [
        "return",
        "high_low",
        "open_close",
        "volume_change",
        "ma5_gap",
        "ma20_gap",
    ]
    features, mean, std = standardize_features(data[feature_columns].to_numpy(dtype=np.float32))
    returns = data["return"].to_numpy(dtype=np.float32)

    x_list = []
    y_list = []
    for i in range(seq_len, len(data) - pred_days + 1):
        x_list.append(features[i - seq_len:i])
        y_list.append(returns[i:i + pred_days])

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return x, y, feature_columns, mean, std


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ReturnTransformer(nn.Module):
    def __init__(self, input_dim, pred_days=PRED_DAYS, d_model=64, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, pred_days),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.position(x)
        encoded = self.encoder(x)
        last_token = encoded[:, -1, :]
        return self.output(last_token)


def split_train_test(x, y, train_ratio=0.8):
    split_index = int(len(x) * train_ratio)
    return x[:split_index], y[:split_index], x[split_index:], y[split_index:]


def train_model(x_train, y_train, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReturnTransformer(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:03d}, train_loss={total_loss / len(dataset):.8f}")

    return model, device


def evaluate_model(model, device, x_test, y_test):
    if len(x_test) == 0:
        return

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test).to(device)).cpu().numpy()

    mae_1d = np.mean(np.abs(pred[:, 0] - y_test[:, 0]))
    mae_3d = np.mean(np.abs(pred.sum(axis=1) - y_test.sum(axis=1)))
    print(f"测试集下一日涨跌幅 MAE: {mae_1d * 100:.4f}%")
    print(f"测试集未来三日累计涨跌幅 MAE: {mae_3d * 100:.4f}%")


def save_model(model, feature_columns, mean, std):
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_columns": feature_columns,
            "feature_mean": mean,
            "feature_std": std,
            "seq_len": SEQ_LEN,
            "pred_days": PRED_DAYS,
        },
        MODEL_FILE,
    )
    print(f"模型已保存到: {MODEL_FILE}")


def predict_latest(model, device, x):
    latest_x = torch.from_numpy(x[-1:]).to(device)
    model.eval()
    with torch.no_grad():
        pred_returns = model(latest_x).cpu().numpy()[0]

    next_1d = pred_returns[0]
    next_3d = pred_returns.sum()
    return next_1d, next_3d, pred_returns


def main():
    print(f"拉取 {TS_CODE} 近 {YEARS} 年数据")
    df = get_index_data()
    data = build_features(df)
    x, y, feature_columns, mean, std = make_dataset(data)

    if len(x) < 200:
        raise ValueError("训练样本太少，无法训练 Transformer")

    x_train, y_train, x_test, y_test = split_train_test(x, y)
    print(f"训练样本: {len(x_train)}, 测试样本: {len(x_test)}")

    model, device = train_model(x_train, y_train, input_dim=x.shape[2])
    evaluate_model(model, device, x_test, y_test)
    save_model(model, feature_columns, mean, std)

    next_1d, next_3d, daily_preds = predict_latest(model, device, x)
    print("\n预测结果：")
    print(f"下一日涨跌幅预测: {next_1d * 100:.4f}%")
    print(f"未来三日累计涨跌幅预测: {next_3d * 100:.4f}%")
    print(
        "未来三日逐日预测: "
        + ", ".join(f"第{i + 1}日 {value * 100:.4f}%" for i, value in enumerate(daily_preds))
    )


if __name__ == "__main__":
    main()
