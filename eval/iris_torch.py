#!/usr/bin/env python3
import os
import json
import random

import numpy as np
import requests
from tqdm import trange
from torch import tensor
from torch.optim import Adam
from torch.nn.functional import softmax, cross_entropy

from minigrad.util import one_hot_encode
from minigrad.tensor import Tensor

np.random.seed(10)

BASE = os.path.dirname(__file__) + "/tmp"

if not os.path.exists(BASE):
  os.makedirs(BASE)

url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
fname = "iris.txt"


def fetch(url):
  fpath = f"{BASE}/{fname}"
  if os.path.exists(fpath):
    return json.loads(open(fpath).read())
  res = requests.get(url).text.strip().split()
  with open(fpath, "w") as f:
    f.write(json.dumps(res))
  return res


def prepare(dataset, test_size=0.2, shuffle=True):
  def parse(dataset):
    x, y = [], []
    for t in dataset:
      t = t.split(",")
      x.append([float(x) for x in t[:-1]])
      y.append(t[-1])
    return x, one_hot_encode(y)

  if shuffle:
    random.shuffle(dataset)

  dataset = parse(dataset)

  i = int((1 - test_size) * len(dataset[0]))

  out = [dataset[0][:i], dataset[1][:i], dataset[0][i:], dataset[1][i:]]

  return tuple(map(lambda x: np.array(x, dtype=np.float32), out))


class NN:
  def __init__(self):
    self.l1: tensor = tensor(Tensor.uniform(4, 128).numpy(), requires_grad=True)
    self.l2: tensor = tensor(Tensor.uniform(128, 64).numpy(), requires_grad=True)
    self.l3: tensor = tensor(Tensor.uniform(64, 3).numpy(), requires_grad=True)

    self.params = [self.l1, self.l2, self.l3]

  def forward(self, x):
    x = x.matmul(self.l1)
    x = x.relu()
    x = x.matmul(self.l2)
    x = x.relu()
    x = x.matmul(self.l3)
    x = softmax(x, dim=-1)
    return x

  def __call__(self, x):
    return self.forward(x)


if __name__ == "__main__":
  dataset = fetch(url)[1:]

  X_train, Y_train, X_test, Y_test = prepare(dataset)

  model = NN()
  optim = Adam(params=model.params, lr=0.001)

  epochs = 10000
  batch_size = 32

  for _ in (t := trange(epochs)):
    optim.zero_grad()

    idx = np.random.choice(len(X_train), batch_size, replace=False)

    x, y = tensor(X_train[idx], requires_grad=True), tensor(Y_train[idx], requires_grad=True)

    out = model(x)

    loss = cross_entropy(out, y)

    loss.backward()

    optim.step()

    # eval
    cat = np.argmax(out.detach().numpy(), axis=-1)
    accuracy = (cat == np.argmax(y.detach().numpy(), axis=-1)).mean()
    t.set_description(f"loss: {loss:.2f} accuracy: {accuracy:.2f}")

  x, y = tensor(X_test), tensor(Y_test)

  out = model(x)

  print("*" * 100)
  cat = np.argmax(out.detach().numpy(), axis=-1)
  accuracy = (cat == np.argmax(y.detach().numpy(), axis=-1)).mean()
  print(f"test accuracy: {accuracy:.2f}")
  print("*" * 100)
