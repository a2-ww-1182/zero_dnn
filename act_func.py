# 活性化関数のまとめ
import numpy as np


def sigmoid(x):  # シグモイド関数
    return 1 / (1 + np.exp(-x))


def identity_function(x):  # 恒等写像
    return x


def softmax(x):  # ソフトマックス、最大値でのスケーリング
    c = np.max(x, axis=-1, keepdims=True)
    a = np.exp(x - c)
    return a / np.sum(a, axis=-1, keepdims=True)
