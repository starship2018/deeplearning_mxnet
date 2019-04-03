import os
import shutil

def reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行
        lines = f.readlines()[1:]
        # rstrip()去掉字符串末尾指定的字符，默认为空格
        # split() 以‘,’为分隔符拆分字符串，并返回一个字符串列表
        tokens = [l.rstrip().split(',') for l in lines]
        # 将列表转化为【字典类型】key_value
        idx_label = dict(((int(idx), label) for idx, label in tokens))
    # set函数创建了一个无限不重复标签集，可迭代 50000个元素
    labels = set(idx_label.values())


    sum_train_valid = len(os.listdir(os.path.join(data_dir, train_dir)))
    # 根据测试验证比，从训练集中进行划分
    n_train = int(sum_train_valid * ( 1 - valid_ratio))
    # assert表示断言，若条件在assert这里是成立的，那么程序照常执行，如果不成立，那么程序立即终止
    assert 0 < n_train < sum_train_valid

    n_train_per_label = n_train // len(labels)
    label_count = {}

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 整理训练集和验证集

    # 这里拿到的train_file是陈列在train文件夹中的所有train样本的文件名 由序号和.png构成
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        #对每一个样本执行如下的操作！

        # 成功取到了样本的序号，并强转为int类型的数据
        idx = int(train_file.split('.')[0])
        # 根据样本序号在标签字典集中取到了起对应的标签数据（分类数据）
        label = idx_label[idx]
        # 在输入目录train_valid_test中创建train_valid再创建以这个样本标签为名的文件夹
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        # 将该样本复制到刚刚新创建出来的目录中去
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))

        if label not in label_count or label_count[label] < n_train_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))


    # 整理测试集
    mkdir_if_not_exist([data_dir, input_dir,'test','unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))

train_dir = 'train'
test_dir = 'test'
batch_size = 1
data_dir = './data/kaggle_cifar10'
label_file = 'trainLabels.csv'
input_dir = 'train-valid_test'
ratio = 0.1
reorg_cifar10_data(data_dir,label_file, train_dir, test_dir, input_dir, ratio)

