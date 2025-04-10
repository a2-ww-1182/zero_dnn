import numpy as np
import act_func

# 入力、第1層の重みとバイアス
X = np.array([1.0, 0.5])
W1 = np.array(([0.1, 0.3, 0.5], [0.2, 0.4, 0.6]))
B1 = np.array([0.1, 0.2, 0.3])
# 第2層の重みとバイアス
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = [0.1, 0.2]
# 第3層の重みとバイアス
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A1 = np.dot(X, W1) + B1
Z1 = act_func.sigmoid(A1)
A2 = np.dot(Z1, W2) + B2
Z2 = act_func.sigmoid(A2)
A3 = np.dot(Z2, W3) + B3
Y = act_func.identity_function(A3)

print(A1)
print(Z1)
print(A2)
print(Z2)
print(A3)
print(Y)
