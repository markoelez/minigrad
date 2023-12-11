#!/usr/bin/env python3
import os
import gzip
import requests
import numpy as np
import random
import json
from minigrad.tensor import Tensor
from minigrad.optim import SGD
from tqdm import trange

np.random.seed(10)

BASE = os.path.dirname(__file__) + '/eval'

if not os.path.exists(BASE): os.makedirs(BASE)

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
fname = 'iris.txt'


def fetch(url):
    fpath = f"{BASE}/{fname}"
    if os.path.exists(fpath):
        return json.loads(open(fpath).read())
    res = requests.get(url).text.strip().split()
    with open(fpath, 'w') as f:
        f.write(json.dumps(res))
    return res


def one_hot_encode(labels):

    m = np.unique(labels)
    h = {x: i for i, x in enumerate(m)}

    a = np.zeros((len(labels), len(h)))
    for i, x in enumerate(labels):
        a[i][h[x]] = 1
    return a


def prepare(dataset, test_size=0.2, shuffle=False):

    def parse(dataset):
        x, y = [], []
        for t in dataset:
            t = t.split(',')
            x.append([float(x) for x in t[:-1]])
            y.append(t[-1])
        return x, one_hot_encode(y)

    if shuffle: random.shuffle(dataset)

    dataset = parse(dataset)

    i = int((1 - test_size) * len(dataset[0]))

    return dataset[0][:i], dataset[1][:i], dataset[0][i:], dataset[1][i:]


class NN:
    def __init__(self):
        self.l1: Tensor = Tensor.uniform(4, 10)
        self.l2: Tensor = Tensor.uniform(10, 3)

    def forward(self, x):
        x = x.dot(self.l1)
        x = x.relu()
        x = x.dot(self.l2)
        x = x.softmax()
        return x

    def __call__(self, x):
        return self.forward(x)


if __name__ == '__main__':

    dataset = fetch(url)[1:]

    X_train, Y_train, X_test, Y_test = prepare(dataset)

    model = NN()
    optimizer = SGD(params=[model.l1, model.l2])

    for _ in (t := trange(1)):
        optimizer.zero_grad()

        x, y = Tensor(X_train), Tensor(Y_train)

        # output = logits
        out = model(x)

        loss = out.cross_entropy(y)

        loss.backward()

        print(out.numpy())
        print(loss)

        # optimizer.step()

        # eval
        cat = np.argmax(out.numpy(), axis=-1)
        accuracy = (cat == np.argmax(y.numpy(), axis=-1)).mean()
        loss = 1
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))