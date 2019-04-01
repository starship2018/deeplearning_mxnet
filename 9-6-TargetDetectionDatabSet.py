from mxnet import gluon, image
from mxnet.gluon import utils as gutils
import sys
import os

sys.path.insert(0,'..')


# 【下载数据集】
def download_pikaku(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8','train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
                    'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        gutils.download(root_url + k, os.path.join(data_dir, k), sha1_hash=v)

#【读取数据集】
def load_data_pikaku(batch_size, edg_size=256):
    data_dir = './data/pikaku'
    download_pikaku(data_dir)
    train_iter = image.ImageDetIter(path_imgrec=os.path.join(data_dir,'train.rec'),
                                    path_imgidx=os.path.join(data_dir,'train.idx'),
                                    batch_size=batch_size,
                                    data_shape=(3, edg_size, edg_size),
                                    shuffle=True,
                                    rand_crop=1,
                                    min_object_covered=0.95,
                                    max_attempts=200)
    val_iter = image.ImageDetIter(path_imgrec=os.path.join(data_dir,'val.rec'),batch_size=batch_size,
                                  data_shape=(3,edg_size,edg_size),shuffle=False)
    return train_iter, val_iter

batch_size, edg_size = 32, 256
train_iter, _ = load_data_pikaku(batch_size,edg_size)
batch = train_iter.next()
print(batch.data[0].shape,batch.label.shape)