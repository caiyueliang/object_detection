from mxnet import gluon
from mxnet import image
from mxnet import nd
# %matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt



# 下载数据
def download_data(data_dir=None):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/')
    if data_dir is not None:
        data_dir = 'data/pikachu/'
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}

    for k, v in dataset.items():
        gluon.utils.download(root_url + k, data_dir + k, sha1_hash=v)


# 读取数据集
# 使用image.ImageDetIter来读取数据。这是针对物体检测的迭代器，(Det表示Detection)。
# 它跟image.ImageIter使用很类似。主要不同是它返回的标号不是单个图片标号，而是每个图片里所有物体的标号，以及其对用的边框。
def get_iterators(data_dir, data_shape, batch_size):
    class_names = ['pikachu']
    num_class = len(class_names)
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir + 'train.rec',
        path_imgidx=data_dir + 'train.idx',
        shuffle=True,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    val_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir + 'val.rec',
        shuffle=False,
        mean=True)
    return train_iter, val_iter, class_names, num_class


def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        fill=False, edgecolor=color, linewidth=linewidth)

# ======================================================================================================================
data_path = 'data/pikachu/'
data_shape = 256
batch_size = 32
rgb_mean = nd.array([123, 117, 104])

if __name__ == '__main__':
    # download_data(data_path)
    train_data, test_data, class_names, num_class = get_iterators(data_path, data_shape, batch_size)

    batch = train_data.next()
    print(batch)



    _, figs = plt.subplots(3, 3, figsize=(6, 6))
    for i in range(3):
        for j in range(3):
            img, labels = batch.data[0][3 * i + j], batch.label[0][3 * i + j]
            # (3L, 256L, 256L) => (256L, 256L, 3L)
            img = img.transpose((1, 2, 0)) + rgb_mean
            img = img.clip(0, 255).asnumpy() / 255
            fig = figs[i][j]
            fig.imshow(img)
            for label in labels:
                rect = box_to_rect(label[1:5] * data_shape, 'red', 2)
                fig.add_patch(rect)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
    plt.show()