# ニューラルネットワーク各層の実装
import numpy as np
from act_func import softmax
from err_func import cross_entropy_error
import os
import sys
sys.path.append(os.pardir)
from im2col import im2col
from col2im import col2im


class Relu:  # ReLu関数
    def __init__(self):
        # True/Falseからなる配列。0以下の要素をTrue,それ以外をFalse
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:  # シグモイド関数
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:  # ニューラルネットワーク順伝播
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.original_x_shape = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(self.original_x_shape)
        return dx


class BatchNorm:  # Batch Normalizationレイヤ
    def __init__(self, r, b):
        self.ub = None  # mean
        self.sigma = None  # var
        self.denom = None  # 1 / sqrt(sigma)
        self.x_prime = None  # (x - mean) / sqrt(sigma)
        self.x_subm = None  # x - mean
        self.x_sq = None  # x_subm ** 2
        self.rt_sigma = None  # sqrt(sigma)
        self.r = r  # gamma
        self.b = b  # beta

        self.dr = None  # dgamma
        self.db = None  # dbeta

    def forward(self, x):
        self.x = x
        self.ub = np.mean(x, axis=0)
        self.x_subm = self.x - self.ub
        self.x_sq = np.square(self.x_subm)
        self.sigma = np.mean(self.x_sq, axis=0)
        self.rt_sigma = np.sqrt(self.sigma + 10e-7)
        self.denom = 1 / self.rt_sigma
        self.x_prime = self.x * self.denom

        out = self.x_prime * self.r + self.b

        return out

    def backward(self, dout):
        self.db = np.sum(dout, axis=0)
        self.dr = np.sum(self.x_prime * dout, axis=0)
        dx_prime = self.r * dout
        dx_subm1 = dx_prime * self.denom
        ddenom = np.sum(dx_prime * self.x_subm, axis=0)
        drt_sigma = -np.square(self.denom) * ddenom
        dsigma = 1 / (2 * self.rt_sigma) * drt_sigma
        dx_sq = dsigma * np.ones_like(self.x) / self.x.shape[0]
        dx_subm2 = 2 * self.x_subm * dx_sq
        dub = -1 * np.sum(dx_subm1 + dx_subm2, axis=0)
        dx1 = dx_subm1 + dx_subm2
        dx2 = dub * np.ones_like(self.x) / self.x.shape[0]

        dx = dx1 + dx2

        return dx


class SoftmaxWithLoss:  # 活性化関数がソフトマックスの出力層兼交差エントロピー誤差の導出層
    def __init__(self):
        self.loss = None  # 損失
        self.y = None  # softmaxの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


class Convolution:  # 畳み込み層
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # フィルタ
        self.b = b  # バイアス
        self.stride = stride  # ストライド
        self.pad = pad  # パディング
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        dcol_W = np.dot(self.col.T, dout)
        self.dW = dcol_W.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:  # プーリング層
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] \
            = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape,
                    self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
