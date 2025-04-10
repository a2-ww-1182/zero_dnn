import numpy as np

# パーセプトロンによるAND実装
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7  # 各種パラメータの設定
    tmp = x1 * w1 + x2 * w2  # 重みづけ計算
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

# NANDの実装
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# ORの実装
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# XORの実装。上で作ったパーセプトロンを組み合わせる。
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))