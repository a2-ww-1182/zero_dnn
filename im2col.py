# CNN,入力データの成型im2colの実装
import numpy as np


def im2col(input_data, filter_h, filter_w, stride, pad):
    """
    input_data: 4次元shapeの入力データ
    filter_h: フィルタの高さ
    filter_w: フィルタの幅
    stride: ストライド
    pad: パディング
    """

    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 通常の畳み込み出力の縦の大きさ
    out_w = (W + 2 * pad - filter_w) // stride + 1  # 通常の畳み込み出力の横の大きさ

    # 入力をpadで指定された分パディング
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
