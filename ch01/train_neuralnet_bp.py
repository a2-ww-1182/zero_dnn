import numpy as np
import sys
import os
sys.path.append(os.pardir)
from mnist import load_mnist
import optimizer
from two_layer_net_bp import TwoLayerNet
import matplotlib.pyplot as plt
# 誤差逆伝播によるMNIST

# mnistのロード
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# 損失関数の記録用
train_loss_list = []
# 訓練時の正解率記録
train_acc_list = []
# 検証時の正解率記録
test_acc_list = []

# ハイパーパラメータ
iters_num = 10000  # 学習の回数
train_size = x_train.shape[0]
batch_size = 100  # ミニバッチのサイズ
learning_rate = 0.1  # 学習率

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
opt = optimizer.Adam()

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算(誤差逆伝播法)
    grad = network.gradient(x_batch, t_batch)

    # 重みの更新
    opt.update(network.params, grad)
    # for key in ('W1', 'b1', 'W2', 'b2'):
    #     network.params[key] -= learning_rate * grad[key]

    # 経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # print(train_acc, test_acc)

# 結果の表示
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(test_acc_list, label='test_acc')
ax.plot(train_acc_list, label='train_acc')
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
plt.tight_layout()
plt.show()
