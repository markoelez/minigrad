#!/usr/bin/env python3
import os
import gzip
import requests
import tarfile
import pathlib
import pickle
import numpy as np
from minigrad.util import one_hot_encode
from torch.optim import SGD, Adam
from torch import tensor
from torch import nn
from torch.nn.functional import softmax, cross_entropy
from tqdm import trange


np.random.seed(10)

BASE = os.path.dirname(__file__) + '/tmp'
if not os.path.exists(BASE): os.makedirs(BASE)

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
files = ['cifar-10-batches-bin/data_batch_1.bin',
         'cifar-10-batches-bin/data_batch_2.bin',
         'cifar-10-batches-bin/data_batch_3.bin',
         'cifar-10-batches-bin/data_batch_4.bin',
         'cifar-10-batches-bin/data_batch_5.bin',
         'cifar-10-batches-bin/test_batch.bin']
out = f'{BASE}/cifar.tar.gz'


def fetch(url: str, out: str, chunk_size: int = 128) -> pathlib.Path:
    fp = pathlib.Path(out)
    if fp.exists(): return fp
    res = requests.get(url, stream=True)
    with open(fp, 'wb') as fd:
        for dat in res.iter_content(chunk_size=chunk_size):
            fd.write(dat)
    return fp


def rgb_to_grayscale(a):
    return np.dot(a[..., :3], [0.2989, 0.5870, 0.1140])


def load(test_size=0.3, grayscale=False):
    fp = fetch(url, out)
    tf = tarfile.open(fp, mode='r:gz')

    fsize = 10000 * (32 * 32 * 3) + 10000
    buff = np.zeros(fsize * 6, dtype='uint8')
    a = sorted([x for x in tf if x .name in files], key=lambda x: x.name)
    for i, fn in enumerate(a):
        f = tf.extractfile(fn)
        buff[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    labels = buff[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buff, np.arange(0, buff.size, 3073))
    # images = pixels.reshape(-1, 3072).astype('float32') / 255
    images = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255

    i = int((1 - test_size) * labels.shape[0])

    X_train, X_test = images[:i], images[i:]
    Y_train, Y_test = labels[:i], labels[i:]

    if grayscale:
        X_train = rgb_to_grayscale(X_train)
        X_test = rgb_to_grayscale(X_test)

    return X_train, one_hot_encode(Y_train), X_test, one_hot_encode(Y_test)


class ConvNet:
    def __init__(self):

        self.c1 = nn.Conv2d(3, 6, 5)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.params = [self.c1, self.c2, self.fc1, self.fc2, self.fc3]

    def forward(self, x, verbose=True):
        if verbose: print('1', x.shape)
        x = x.reshape(shape=(-1, 3, 32, 32))
        if verbose: print('2', x.shape)
        x = x.conv2d(self.c1).relu().max_pool2d()
        if verbose: print('3', x.shape)
        x = x.conv2d(self.c2).relu().max_pool2d()
        if verbose: print('4', x.shape)
        x = x.reshape(shape=[x.shape[0], -1])
        if verbose: print('5', x.shape)
        x = x.dot(self.l1).relu()
        if verbose: print('6', x.shape)
        x = x.dot(self.l2)
        if verbose: print('7', x.shape)
        x = x.softmax()
        if verbose: print('8', x.shape)
        return x

    def __call__(self, x):
        return self.forward(x)


if __name__ == '__main__':

    if not os.path.exists(out): fetch_all()

    X_train, Y_train, X_test, Y_test = load()

    model = ConvNet()
    optim = Adam(params=model.params, lr=0.001)

    epochs = 500
    batch_size = 128

    for _ in (t := trange(epochs)):

        idx = np.random.choice(len(X_train), batch_size, replace=False)

        x, y = Tensor(X_train[idx]), Tensor(Y_train[idx])

        out = model(x)

        loss = out.cross_entropy(y)

        optim.zero_grad()

        loss.backward()

        optim.step()

        cat = np.argmax(out.numpy(), axis=-1)
        accuracy = (cat == np.argmax(y.numpy(), axis=-1)).mean()
        t.set_description(f"loss: {loss.data:.2f} accuracy: {accuracy:.2f}")

    x, y = Tensor(X_test), Tensor(Y_test)

    out = model(x)

    print('*' * 100)
    cat = np.argmax(out.numpy(), axis=-1)
    accuracy = (cat == np.argmax(y.numpy(), axis=-1)).mean()
    print(f"test accuracy: {accuracy:.2f}")
    print('*' * 100)
