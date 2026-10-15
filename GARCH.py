"""
GARCH 波动率模型教学脚本。

核心思想：
1. 收益率本身通常难以预测。
2. 但收益率的“波动大小”常常有聚集性：大波动后更容易继续大波动。
3. GARCH 模型就是用过去的冲击和过去的波动率，预测今天的条件波动率。

GARCH(1,1) 方差方程：
    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

含义：
    omega: 长期基础波动
    alpha: 昨天突发冲击对今天波动率的影响
    beta : 昨天波动率惯性对今天波动率的影响
    alpha + beta 越接近 1，说明波动率持续性越强
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

import config


TS_CODE = "000300.SH"  # 沪深300指数
START_DATE = "20150101"


def get_index_data(ts_code=TS_CODE, start_date=START_DATE):
    """拉取指数日线，并按日期升序返回。"""
    ts.set_token(config.TUSHARE_TOKEN)
    df = ts.pro_bar(ts_code=ts_code, asset="I", start_date=start_date)

    if df is None or df.empty:
        raise ValueError("没有拉取到指数数据，请检查网络、Tushare token 或 ts_code")

    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def calculate_log_returns(df):
    """计算对数收益率，乘以 100 后单位近似为百分比。"""
    returns = 100 * np.log(df["close"] / df["close"].shift(1))
    returns.name = "log_return"
    return returns.dropna()


def explain_returns(returns):
    """用统计检验说明为什么收益率适合做 GARCH 建模。"""
    print("\n========== 1. 收益率序列 ==========")
    print("GARCH 不直接预测价格，而是建模收益率的波动率。")
    print(f"样本数量: {len(returns)}")
    print(f"平均日收益率: {returns.mean():.4f}%")
    print(f"收益率标准差: {returns.std():.4f}%")
    print(f"最大单日涨幅: {returns.max():.4f}%")
    print(f"最大单日跌幅: {returns.min():.4f}%")

    adf_stat, adf_pvalue, *_ = adfuller(returns)
    print("\nADF 平稳性检验：")
    print(f"ADF statistic = {adf_stat:.4f}, p-value = {adf_pvalue:.6f}")
    if adf_pvalue < 0.05:
        print("结论: 收益率序列通常是平稳的，可以用于 GARCH 建模。")
    else:
        print("结论: 收益率序列可能不平稳，需要谨慎处理。")


def check_arch_effect(returns):
    """检查平方收益率是否存在自相关，也就是 ARCH 效应。"""
    print("\n========== 2. ARCH 效应 ==========")
    print("如果平方收益率存在自相关，说明大波动会成簇出现，GARCH 才有意义。")

    lb_result = acorr_ljungbox(returns**2, lags=[10], return_df=True)
    pvalue = lb_result["lb_pvalue"].iloc[0]
    print(f"Ljung-Box 检验 p-value = {pvalue:.6f}")

    if pvalue < 0.05:
        print("结论: 存在显著 ARCH 效应，适合使用 GARCH 模型。")
    else:
        print("结论: ARCH 效应不明显，GARCH 的解释力可能有限。")


def fit_garch_11(returns):
    """拟合最经典的 GARCH(1,1) 模型。"""
    print("\n========== 3. 拟合 GARCH(1,1) ==========")
    print("模型形式: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2")

    model = arch_model(
        returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )
    result = model.fit(disp="off")
    print(result.summary())
    return result


def explain_garch_params(result):
    """把模型参数翻译成更容易理解的语言。"""
    print("\n========== 4. 参数怎么理解 ==========")
    params = result.params
    omega = params.get("omega", np.nan)
    alpha = params.get("alpha[1]", np.nan)
    beta = params.get("beta[1]", np.nan)
    persistence = alpha + beta

    print(f"omega = {omega:.6f}: 长期基础波动水平。")
    print(f"alpha = {alpha:.6f}: 昨天冲击对今天波动率的影响。")
    print(f"beta  = {beta:.6f}: 昨天波动率延续到今天的程度。")
    print(f"alpha + beta = {persistence:.6f}: 波动率持续性。")

    if persistence > 0.95:
        print("解释: 波动率持续性很强，市场从高波动恢复到低波动会比较慢。")
    elif persistence > 0.8:
        print("解释: 波动率有明显持续性，但衰减速度相对可控。")
    else:
        print("解释: 波动率持续性不强，冲击衰减较快。")


def forecast_next_day_volatility(result):
    """预测下一交易日条件波动率。"""
    print("\n========== 5. 下一交易日波动率预测 ==========")
    forecast = result.forecast(horizon=1)
    variance = forecast.variance.iloc[-1, 0]
    volatility = np.sqrt(variance)
    print(f"下一交易日预测波动率: {volatility:.4f}%")
    return volatility


def plot_returns_and_volatility(df, returns, result):
    """画出收益率和 GARCH 条件波动率，直观看波动聚集。"""
    plot_df = df.iloc[1:].copy()
    plot_df["return"] = returns.values
    plot_df["volatility"] = result.conditional_volatility.values

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(plot_df["trade_date"], plot_df["return"], color="steelblue", alpha=0.55, label="日对数收益率")
    ax1.set_ylabel("收益率 (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(plot_df["trade_date"], plot_df["volatility"], color="crimson", linewidth=1.4, label="GARCH条件波动率")
    ax2.set_ylabel("条件波动率 (%)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    plt.title("沪深300收益率与 GARCH(1,1) 条件波动率")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    print("开始 GARCH 波动率模型教学示例。")
    df = get_index_data()
    returns = calculate_log_returns(df)

    explain_returns(returns)
    check_arch_effect(returns)
    result = fit_garch_11(returns)
    explain_garch_params(result)
    forecast_next_day_volatility(result)
    plot_returns_and_volatility(df, returns, result)


if __name__ == "__main__":
    main()
