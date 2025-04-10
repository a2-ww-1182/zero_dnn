import numpy as np
from simple_conv_net import SimpleConvNet
import sys
import os
sys.path.append(os.pardir)
from mnist import load_mnist
import matplotlib.pyplot as plt
import optimizer


(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True, flatten=False)

train_loss_list = []

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = SimpleConvNet()
opt = optimizer.Adam()

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.gradient(x_batch, t_batch)

    # 重みの更新
    opt.update(network.params, grad)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(loss)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(train_loss_list)
ax.set_xlabel("train_num")
ax.set_ylabel("loss")
plt.tight_layout()
plt.show()
