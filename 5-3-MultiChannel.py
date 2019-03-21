from mxnet import nd
from mxnet.gluon import nn

def corr2d(X,K):
    h ,w = K.shape
    # 卷积运算中，卷积结果的计算公式： (n + 2p - f) / s + 1
    # n是原矩阵的高度，p是padding填充数，f是卷积核的高度，s是卷积步数大小。长度和高度一致
    Y = nd.zeros((X.shape[0] - h + 1,X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 这里的互相关运算 直接乘并求和即可得到！
            # 这里的i:i+h 和 j:j+w 要仔细揣摩！
            Y[i,j] = (X[i:i + h ,j:j + w] * K).sum()
    return Y

def corr2d_multi_in(X,K):
    return nd.add_n(*[corr2d(x,k) for x,k in zip(X,K)])


X = nd.array([[[0,1,2],[3,4,5],[6,7,8]],[[1,2,3],[4,5,6],[7,8,9]]])

K = nd.array([[[0,1],[2,3]],[[1,2],[3,4]]])

print(corr2d_multi_in(X,K))