# 計算グラフの実装


class MulLayer:  # 計算グラフ中の乗算ノードの実装
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):  # 順伝播,乗算
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):  # 逆伝播,微分
        """dout: 上流から伝わってきた微分"""
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:  # 計算グラフ中の加算ノードの実装
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
