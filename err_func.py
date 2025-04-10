# 損失関数を定義するコード
import numpy as np


def sum_squared_error(y, t):  # 二乗和誤差
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):  # 交差エントロピー誤差
    if y.ndim == 1:  # y, tが1次元の場合は2次元に変換
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    delta = 1e-7  # 計算不可能防止のための微小量
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
