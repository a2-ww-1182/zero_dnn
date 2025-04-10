# 5章例題リンゴ購入の実装
from layer_naive import MulLayer

apple = 100  # リンゴの値段
apple_num = 2  # リンゴの個数
tax = 1.1  # 消費税

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

# backward
dprice = 1  # 全計算結果の値段に関する微分
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)
