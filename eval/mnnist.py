#!/usr/bin/env python3
import os
import gzip
import requests
import numpy as np
from minigrad.tensor import Tensor
from minigrad.nn.optim import SGD
from tqdm import trange

BASE = os.path.dirname(__file__) + '/eval'

if not os.path.exists(BASE): os.makedirs(BASE)

dataset_fnames = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download(url, out, chunk_size=128):
    res = requests.get(url, stream=True)
    with open(out, 'wb') as fd:
        for dat in res.iter_content(chunk_size=chunk_size):
            fd.write(dat)


def download_all():
    for fname in dataset_fnames:
        url = f"http://yann.lecun.com/exdb/mnist/{fname}"
        out = f"{BASE}/{fname}"
        download(url, out)


def load():
    def parse(file): return np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
    X_train = parse(f"{BASE}/{dataset_fnames[0]}")[0x10:].reshape((-1, 28 * 28)).astype(np.float32)
    Y_train = parse(f"{BASE}/{dataset_fnames[1]}")[8:]
    X_test = parse(f"{BASE}/{dataset_fnames[2]}")[0x10:].reshape((-1, 28 * 28)).astype(np.float32)
    Y_test = parse(f"{BASE}/{dataset_fnames[3]}")[8:]
    return X_train, Y_train, X_test, Y_test


def train():
    pass


def train(model, X_train, Y_train, optim, epochs, BS=128, lossfn=lambda out, y: out.sparse_categorical_crossentropy(y)):
    pass


class ConvNet:
    def __init__(self):
        pass

    def forward(self):
        pass


if __name__ == '__main__':

    if not all(os.path.exists(f"{BASE}/{fname}") for fname in dataset_fnames):
        download_all()

    X_train, Y_train, X_test, Y_test = load()

    model = ConvNet()
    optimizer = SGD()

    train(model, X_train, Y_train, optimizer, 2)
