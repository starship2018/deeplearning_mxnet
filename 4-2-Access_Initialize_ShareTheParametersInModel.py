from mxnet import init,nd

from mxnet.gluon import nn

net = nn.Sequential()

net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))

# 使用默认的方法初始化模型参数

# initialize方法默认初始化结果为：权重参数元素为[-0.07,0.07]之间均匀分布的随机数
# 偏差参数全部为0
net.initialize()


# python中函数参数值一般要按照顺序进行传递，若要跳过，一定要注明参数名【要插队，先报名】
# 在定义函数时，param = value就是默认值！

x = nd.random.uniform(shape = (2,20))
y = net(x)

# print(net[0].params, type(net[0].params))
#
# print(net[0].params['dense0_weight'],net[0].weight)

# print(net[0].weight.data())

# print(net[0].weight.grad())

# 获取net中包含的所有函数
# print(net.collect_params())

# collect-params提供了正则表达式来筛选所需的参数
# print(net.collect_params('.*bias'))

net.initialize(init.Normal(sigma=0.01),force_reinit=True)
# init.Normal(sigma=0.01) 表示均值为0，标准差为0.01的正态分布随机数
# force_reinit 强制重新初始化


net.initialize(init.Constant(2),force_reinit=True)
# 使用常数来初始化权重参数

# 【自定义参数初始化函数】

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init',name,data.shape)
        data[:] = nd.random.uniform(-10,10,data.shape)
        data *= data.abs() >= 5





net.initialize(MyInit(),force_reinit=True)

# 【直接修改模型参数】

net[0].weight.set_data(net[0].weight.data()+1)







print(net[0].weight.data()[0])

