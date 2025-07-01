import os
import gzip

import numpy as np
import requests
from tqdm import trange

from minigrad.util import one_hot_encode
from minigrad.optim import Adam
from minigrad.tensor import Tensor

np.random.seed(10)

BASE = os.path.dirname(__file__) + "/tmp"
if not os.path.exists(BASE):
  os.makedirs(BASE)

dataset_fnames = [
  "train-images-idx3-ubyte.gz",
  "train-labels-idx1-ubyte.gz",
  "t10k-images-idx3-ubyte.gz",
  "t10k-labels-idx1-ubyte.gz",
]


def fetch(url: str, out: str, chunk_size: int = 128):
  res = requests.get(url, stream=True)
  with open(out, "wb") as fd:
    for dat in res.iter_content(chunk_size=chunk_size):
      fd.write(dat)


def fetch_all():
  for fname in dataset_fnames:
    url = f"http://yann.lecun.com/exdb/mnist/{fname}"
    out = f"{BASE}/{fname}"
    fetch(url, out)


def load():
  def parse(file):
    return np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()

  X_train = parse(f"{BASE}/{dataset_fnames[0]}")[0x10:].reshape((-1, 28 * 28)).astype(np.float32)
  Y_train = parse(f"{BASE}/{dataset_fnames[1]}")[8:]
  X_test = parse(f"{BASE}/{dataset_fnames[2]}")[0x10:].reshape((-1, 28 * 28)).astype(np.float32)
  Y_test = parse(f"{BASE}/{dataset_fnames[3]}")[8:]
  return X_train, one_hot_encode(Y_train), X_test, one_hot_encode(Y_test)


class ConvNet:
  def __init__(self):
    conv, cin, cout = 3, 8, 16

    self.c1 = Tensor.uniform(cin, 1, conv, conv)
    self.c2 = Tensor.uniform(cout, cin, conv, conv)
    self.l1 = Tensor.uniform(cout * 5 * 5, 10)

    self.params = [self.l1, self.c1, self.c2]

  def forward(self, x):
    x = x.reshape(shape=(-1, 1, 28, 28))
    x = x.conv2d(self.c1).relu().max_pool2d()
    x = x.conv2d(self.c2).relu().max_pool2d()
    x = x.reshape(shape=[x.shape[0], -1])
    x = x.dot(self.l1)
    x = x.softmax()
    return x

  def __call__(self, x):
    return self.forward(x)


class NN:
  def __init__(self):
    self.l1: Tensor = Tensor.uniform(784, 128)
    self.l2: Tensor = Tensor.uniform(128, 10)

    self.params = [self.l1, self.l2]

  def forward(self, x):
    x = x.dot(self.l1)
    x = x.relu()
    x = x.dot(self.l2)
    x = x.softmax()
    return x

  def __call__(self, x):
    return self.forward(x)


if __name__ == "__main__":
  if not all(os.path.exists(f"{BASE}/{fname}") for fname in dataset_fnames):
    fetch_all()

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

  print("*" * 100)
  cat = np.argmax(out.numpy(), axis=-1)
  accuracy = (cat == np.argmax(y.numpy(), axis=-1)).mean()
  print(f"test accuracy: {accuracy:.2f}")
  print("*" * 100)
