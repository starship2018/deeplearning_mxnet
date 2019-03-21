from mxnet import gluon,autograd,init,nd
from mxnet.gluon import data as gdata,loss as gloss,nn
from matplotlib import pyplot as plt

# 【生成数据集】

n_train = 40
n_test = 40
n_inputs = 200

true_w =nd.ones(shape=(n_inputs,1)) * 0.01
true_b = 0.05

features = nd.random.normal(shape=(n_test+n_train,n_inputs))
labels = nd.dot(features,true_w) + true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)

train_features ,test_features = features[:n_train,:], features[n_train:,:]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 【初始化模型参数】
w = nd.random.normal(scale=0.01,shape=(n_inputs,1))

b = nd.zeros(shape=(1,))

w.attach_grad()
b.attach_grad()

# 【定义模型】

def net(X):
    return nd.dot(X,w) + b

# 【定义L2范数惩罚项】

def l2_penalty(w):
    return (w**2).sum() / 2

# 【定义L2范数损失函数】

def loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/ 2

# 【定义优化sgd优化算法】
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# 画图函数
def semilogy(x_vals, y_vals, x_label, y_label,x2_vals =None,y2_vals =None,legend =None):
    plt.semilogy(x_vals, y_vals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals)
        plt.legend(legend)
    plt.show()

# 【训练模型】

batch_size = 1
n_epochs = 100
lr = 0.003

train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels),batch_size,shuffle=True)

def fit_and_plot(lambd):
    train_ls, test_ls = [], []
    for _ in range(n_epochs):
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y) +  l2_penalty(w) * lambd
            l.backward()
            sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features),train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    semilogy(range(1,n_epochs+1),train_ls,'epochs','loss',range(1,n_epochs+1), test_ls,['train','test'])
    print('L2 norm of w',w.norm().asscalar())


def fit_and_plot_gluon(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))

    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd',{'learning_rate':lr,'wd':wd})
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd', {'learning_rate':lr})
    train_ls ,test_ls = [], []
    for _ in range(n_epochs):
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    semilogy(range(1,n_epochs+1), train_ls,'epoch', 'loss', range(1,n_epochs+1), test_ls,
             ['train','test'])
    print('L2 normal of w', net[0].weight.data().norm().asscalar())


    # 关于如何在不使用mxnet的情况下输出权重值w还需要研究！！！！！！

    # print('L2 normal of w', w.norm().asscalar())


# fit_and_plot(lambd=3)

fit_and_plot_gluon(3)





