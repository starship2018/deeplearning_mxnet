from mxnet import nd
from mxnet.gluon import nn


# 池化层的向前运算
def pool2d(X,pool_size,mode = 'max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h +1 , X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j] = X[i:i+p_h,j:j+p_w].max()
            elif mode == 'avg':
                Y[i,j] = X[i:i+p_h,j:j+p_w].mean()
    return Y

# X = nd.array([[1,2,3],[3,4,5],[6,7,8]])
# print(pool2d(X,(2,2)))
# print(pool2d(X,(2,2),mode='avg'))


# 1和1 指的是批量和通道数
X = nd.arange(16).reshape((1,1,4,4))
# print(X)

Pool2d = nn.MaxPool2D(3,padding=1, strides=2) # 指定池化层的大小,若不说明padding和stride 那么默认就也是3！
# 因为池化层没有模型参数，因此并不需要进行参数初始化就可以开始进行计算
# print(Pool2d(X))

X = nd.concat(X,X + 1, dim=1)
Pool2d = nn.MaxPool2D(3,padding=1, strides=2)
print(Pool2d(X))