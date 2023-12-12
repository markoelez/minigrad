#!/usr/bin/env python3
import os
import gzip
import requests
import numpy as np
import random
import json
from minigrad.tensor import Tensor
from minigrad.optim import SGD
from minigrad.util import one_hot_encode
from tqdm import trange

np.random.seed(10)

BASE = os.path.dirname(__file__) + '/tmp'

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

    out = [dataset[0][:i], dataset[1][:i], dataset[0][i:], dataset[1][i:]]

    return tuple(map(np.array, out))


class NN:
    def __init__(self):
        self.l1: Tensor = Tensor.uniform(4, 1000)
        self.l2: Tensor = Tensor.uniform(1000, 500)
        self.l3: Tensor = Tensor.uniform(500, 300)
        self.l4: Tensor = Tensor.uniform(300, 3)

    def forward(self, x):
        x = x.dot(self.l1)
        x = x.relu()
        x = x.dot(self.l2)
        x = x.relu()
        x = x.dot(self.l3)
        x = x.relu()
        x = x.dot(self.l4)
        x = x.softmax()
        return x

    def __call__(self, x):
        return self.forward(x)


if __name__ == '__main__':

    dataset = fetch(url)[1:]

    X_train, Y_train, X_test, Y_test = prepare(dataset)

    model = NN()
    optimizer = SGD(params=[model.l1, model.l2], lr=0.001)

    epochs = 1000
    batch_size = 20

    for _ in (t := trange(epochs)):

        idx = np.random.choice(len(X_train), batch_size, replace=False)

        x, y = Tensor(X_train[idx]), Tensor(Y_train[idx])

        # output = logits
        out = model(x)

        loss = out.cross_entropy(y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # eval
        cat = np.argmax(out.numpy(), axis=-1)
        accuracy = (cat == np.argmax(y.numpy(), axis=-1)).mean()
        t.set_description("loss %.2f accuracy %.2f" % (loss.data, accuracy))
