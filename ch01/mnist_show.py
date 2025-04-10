# mnistデータセットを画像表示するためのコード

import sys
import os
sys.path.append(os.pardir)
from matplotlib import pyplot as plt
from mnist import load_mnist


def img_show(img):  # 画像描画
    plt.imshow(img)  # もとのコードではPIL使ってたけど何かダメだったのでPlt使用
    plt.show()


(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)  # 元のサイズに変形
print(img.shape)

img_show(img)
