#!/usr/bin/env python3
import os
import gzip
import requests
import numpy as np
import random
import json
from torch import tensor
from minigrad.tensor import Tensor
from torch.optim import SGD
from torch.nn.functional import softmax, cross_entropy
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
        self.l1: tensor = tensor(Tensor.uniform(4, 10).numpy(), requires_grad=True)
        self.l2: tensor = tensor(Tensor.uniform(10, 3).numpy(), requires_grad=True)

    def forward(self, x):
        x = x.matmul(self.l1)
        x = x.relu()
        x = x.matmul(self.l2)
        x = softmax(x, dim=-1)
        return x

    def __call__(self, x):
        return self.forward(x)


if __name__ == '__main__':

    dataset = fetch(url)[1:]

    X_train, Y_train, X_test, Y_test = prepare(dataset)

    model = NN()
    optimizer = SGD(params=[model.l1, model.l2], lr=0.01)

    for _ in (t := trange(10000)):

        x, y = tensor(X_train, requires_grad=True), tensor(Y_train, requires_grad=True)

        # output = logits
        out = model(x)

        loss = cross_entropy(out, y)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # eval
        cat = np.argmax(out.detach().numpy(), axis=-1)
        accuracy = (cat == np.argmax(y.detach().numpy(), axis=-1)).mean()
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
