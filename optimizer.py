# パラメータ更新則のまとめ
import numpy as np


class SGD:  # SGD
    def __init__(self, lr=0.01):
        self.lr = lr  # 学習率

    def update(self, params, grads):  # 更新
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:  # Momentum
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr  # 学習率
        self.momentum = momentum  # 運動量
        self.v = None

    def update(self, params, grads):
        if self.v is None:  # v初期化
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:  # AdaGrad
    def __init__(self, lr=0.01):
        self.lr = lr  # 学習率
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += np.square(grads[key])
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:  # Adam
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.t += 1
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + \
                (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + \
                (1 - self.beta2) * np.square(grads[key])
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            params[key] -= self.alpha * m_hat / (np.sqrt(v_hat) + 1e-8)
