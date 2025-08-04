import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL


def calculate_complexity(patch, return_all=False):
    patch = np.asarray(patch).astype(np.float32)
    patch_len = len(patch)

    # 1. Trend Strength
    try:
        res = STL(patch, period=max(patch_len // 2, 2)).fit()
        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid

        deseasonal = patch - seasonal
        var_deseasonal = np.var(deseasonal)
        var_resid = np.var(resid)

        if np.isnan(var_deseasonal) or var_deseasonal == 0:
            trend_strength = 0
        else:
            trend_strength = 1 - (var_resid / var_deseasonal)
            trend_strength = float(np.nan_to_num(trend_strength, nan=0.0, posinf=0.0, neginf=0.0))
    except:
        trend_strength = 0

    # 2. Derivative STD
    diff = np.diff(patch)
    if np.isnan(diff).any() or np.isinf(diff).any():
        std_of_first_derivative = 0
    else:
        std_val = np.std(diff)
        std_val = np.log1p(std_val)  # log 缩放
        std_of_first_derivative = 1 / (1 + np.exp(-(std_val - 1.0)))  # sigmoid 映射到 [0, 1]

    # 3. Autocorrelation
    if patch_len > 1:
        if np.isnan(patch).any() or np.isinf(patch).any() or np.std(patch) == 0:
            autocorr = 0
        else:
            try:
                acf_result = acf(patch, nlags=1, fft=False)
                autocorr = float(np.nan_to_num(np.abs(acf_result[1]), nan=0.0, posinf=0.0, neginf=0.0))
            except:
                autocorr = 0
    else:
        autocorr = 0

    result = np.array([trend_strength, std_of_first_derivative, autocorr], dtype=np.float32)
    result = np.round(result, 3)  # 保留三位小数
    # print(result)
    return result


def calculate_complexity_1(patch):
    # patch: shape [token_len]
    return np.std(patch)  # 或其他复杂度指标