import sys
import os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist

# mnistデータのロード
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
# ミニバッチのサイズ
batch_size = 10
# ミニバッチの抽出
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
