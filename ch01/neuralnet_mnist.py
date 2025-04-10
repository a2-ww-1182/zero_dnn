import sys
import os
import numpy as np
import pickle
sys.path.append(os.pardir)
from mnist import load_mnist
from act_func import sigmoid
from act_func import softmax


def get_data():  # mnistデータセットの読み込み
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():  # 重みとバイアスの読み込み
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):  # 読み込んだ重みとバイアスを元に推論
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


if __name__ == '__main__':
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        y = predict(network, x[i:i+batch_size])
        p = np.argmax(y, axis=1)
        # ターゲットラベルと推論結果が同じであればカウント
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    # 正解率の導出
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
