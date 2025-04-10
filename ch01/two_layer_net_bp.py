# 誤差逆伝播法にもとづく2層ニューラルネットワーク
import sys
import os
import numpy as np
sys.path.append(os.pardir)
from layers import *
from collections import OrderedDict
from diff import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.0001):
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.params['gamma1'] = np.ones_like(self.params['b1'])
        self.params['beta1'] = np.zeros_like(self.params['b1'])

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['BatchNorm1'] = \
            BatchNorm(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):  # 順伝播による推論
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):  # 誤差の導出
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):  # 正解率の導出
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

    def gradient(self, x, t):  # 誤差逆伝播による勾配の導出
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)  # 出力層の逆伝播

        # 隠れ層,入力層の逆伝播
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 求めた勾配を辞書として格納
        # 正則化項を加算
        reg = 1e-4
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW + reg * self.params['W1']
        grads['b1'] = self.layers['Affine1'].db + reg * self.params['b1']
        grads['W2'] = self.layers['Affine2'].dW + reg * self.params['W2']
        grads['b2'] = self.layers['Affine2'].db + reg * self.params['b2']
        grads['gamma1'] = self.layers['BatchNorm1'].dr
        grads['beta1'] = self.layers['BatchNorm1'].db

        return grads
