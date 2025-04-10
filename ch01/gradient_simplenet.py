import sys
import os
sys.path.append(os.pardir)
import numpy as np
from act_func import softmax
from err_func import cross_entropy_error
from diff import numerical_gradient


# 簡単なニューラルネットワークを定義
class simpleNet:
    def __init__(self):
        # 重み行列の初期化
        self.W = np.random.randn(2, 3)

    # 重みと入力の積
    def predict(self, x):
        return np.dot(x, self.W)

    # 交差エントロピーによる損失の導出
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == '__main__':
    net = simpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)

    print(np.argmax(p))

    t = np.array([0, 0, 1])
    print(net.loss(x, t))

    def f(W):
        return net.loss(x, t)

    dw = numerical_gradient(f, net.W)
    print(dw)
