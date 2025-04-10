import numpy as np
import sys
import os
sys.path.append(os.pardir)
from mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt 

# mnistのロード
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# 損失関数の記録用
train_loss_list = []

# ハイパーパラメータ
iters_num = 10000  # 学習の回数
train_size = x_train.shape[0]
batch_size = 100  # ミニバッチのサイズ
learning_rate = 0.01  # 学習率

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)

    # 重みの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

# 結果の表示
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(loss)
ax.set_xlabel("iteration")
ax.set_ylabel("loss")
