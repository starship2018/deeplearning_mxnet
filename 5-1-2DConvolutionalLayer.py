from mxnet import autograd, nd
from mxnet.gluon import nn
import mxnet as mx

# 【手工二维卷积计算】

# ！！！！！手工卷积运算（互相关运算）！！！！！！
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

X = nd.array([[1,2,3],[4,5,6],[7,8,9]])
K = nd.array([[1,2],[3,4]])
# print(corr2d(X, K))

class Conv2D(nn.Block):
    # 在卷积神经网络中，卷积核就相当于权重参数w,都是需要经过大量的训练而得到的
    def __init__(self,kernel_size,**kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight',shape = kernel_size)
        self.bias = self.params.get('bias',shape = (1,))

    def forward(self, x):
        return corr2d(x,self.weight.data()) + self.bias.data()

X = nd.ones((6,8))
X[:,2:6] = 0
# print(X)

# 构造卷积核 ，使得当左右相邻元素相同时，输出为0，若不一样时，输出为1，已达到边缘检测的效果！


# 使用nd或者np构造矩阵时，一定不要忘记写入两个[]，而不是因为是横向量或是竖向量而只写一个[]
K = nd.array([[1,-1]])

Y = corr2d(X,K)
# print(Y)






# 【使用gluon完成卷积运算】
# 构造一个输出通道为1，核形状为（1，2）的二维卷积层

# 定义模型
conv2d = nn.Conv2D(1,kernel_size=(1,2))
# 初始化模型函数
conv2d.initialize()

# 二维卷积层使用4维输入输出，格式为（样本，通道，高，宽），这里的批量大小和通道数均为1

X = X.reshape((1,1,6,8))

Y = Y.reshape((1,1,6,7))

num_epochs = 10
for epoch in range(num_epochs):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    conv2d.weight.data()[:] -= 3/100 * conv2d.weight.grad()
    print('epoch %d  loss %0.3f' % (epoch, l.sum().asscalar()))

print(conv2d.weight.data())