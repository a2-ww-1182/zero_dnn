# 数値微分を実装するコード
import numpy as np


def numerical_diff(f, x):  # 1変数関数の微分
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient_1d(f, x):  # 1変数関数の勾配
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x + h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x - h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 値を元に戻す

    return grad


def numerical_gradient(f, x):  # 多変数関数の勾配
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

    return grad


def gradient_descent(f, init_x, lr=0.01, step_mum=100):  # 勾配降下法
    """lr: 学習率
       step_num: 何回更新するか"""

    x = init_x

    # step_numで指定した回数xを更新
    for i in range(step_mum):
        grad = numerical_gradient(f, x)
        x -= grad * lr

    return x
