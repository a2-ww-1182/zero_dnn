import sys
import os
sys.path.append(os.pardir)
import numpy as np
from diff import numerical_gradient
from act_func import sigmoid, softmax
from err_func import cross_entropy_error

# 2層ニューラルネットワーク
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        """input_size: 入力の次元
           hidden_size: 隠れ層の次元
           output_size: 出力の次元
           weight_init_std: 重みのスケーリング"""

        # 重みの初期化。辞書として実装
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):  # 順伝播
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):  # 損失関数
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):  # 正答率
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):  # 層ごとに分けて勾配の導出
        # 損失関数を重みの関数として定義
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
