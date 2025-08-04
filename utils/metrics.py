# ---------------------------------------------------------------------------------
# Portions of this file are derived from gluonts (mase, msis)
# - Source: https://github.com/awslabs/gluonts
# - Paper: GluonTS: Probabilistic and Neural Time Series Modeling in Python
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------
import torch
import numpy as np
from scipy.stats import t
from gluonts.time_feature import get_seasonality

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


if __name__ == '__main__':
    # 真实值 (假设 true 是一个 numpy 数组)
    past = np.random.rand(32, 192, 7)  # 假设 batch_size=32, 输入长度=192, N=7
    true = np.random.rand(32, 96, 7)  # 假设 batch_size=32, 预测长度=96, N=7

    # 模型预测输出 (NIG 分布参数)
    output = torch.rand(32, 96, 7)  # 预测的均值 μ
    output_v = torch.rand(32, 96, 7) + 1e-6  # 预测的不确定性 ν
    output_alpha = torch.rand(32, 96, 7) + 1.1  # 预测的 alpha 参数
    output_beta = torch.rand(32, 96, 7) + 1e-6  # 预测的 beta 参数

    # 转换到 numpy 格式
    output_np = output.cpu().detach().numpy()
    output_v_np = output_v.cpu().detach().numpy()
    output_alpha_np = output_alpha.cpu().detach().numpy()
    output_beta_np = output_beta.cpu().detach().numpy()

    print(output_np, output_v_np, output_alpha_np, output_beta_np, true)

    mae, mse, rmse, mape, mspe, mase_score, crps_score, msis_score = metric(past, output_np, true, output_np, output_v_np, output_alpha_np, output_beta_np, alpha_level=0.05, model_id = "ETTm2_672_96")

    # **4. 打印结果**
    print(f"mae: {mae:.4f}")
    print(f"mse: {mse:.4f}")
    print(f"rmse: {rmse:.4f}")
    print(f"mape: {mape:.4f}")
    print(f"mspe: {mspe:.4f}")
    print(f"MASE: {mase_score:.4f}")
    print(f"CRPS: {crps_score:.4f}")
    print(f"MSIS: {msis_score:.4f}")
