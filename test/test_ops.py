import pytest
import random
import math
import time
import numpy as np
from minigrad.tensor import Tensor
from torch import tensor
from torch.nn.functional import softmax, cross_entropy


shape = (5, 3)


@pytest.fixture(autouse=True)
def setup():
    np.random.seed(10)


def assert_eq(x, y):
    assert np.allclose(x, y, rtol=1e-09, atol=1e-9)


def test_sum():
    a = np.random.randn(*shape)
    x, y = Tensor(a), tensor(a, requires_grad=True)
    # forward
    xx = x.sum()
    yy = y.sum()
    yy.retain_grad()
    assert_eq(xx.numpy(), yy.detach().numpy())
    assert_eq(x.numpy(), y.detach().numpy())
    # backward
    yy.backward()
    xx.backward()
    assert_eq(xx.grad, yy.grad)
    assert_eq(x.grad, y.grad)


def test_max():
    a = np.random.randn(*shape)
    # axis = -1
    x, y = Tensor(a), tensor(a, requires_grad=True)
    # forward
    xx = x.max(axis=-1)
    yy = y.max(dim=-1)[0]
    yy.retain_grad()
    xxx, yyy = xx.sum(), yy.sum()
    yyy.retain_grad()
    assert_eq(xxx.numpy(), yyy.detach().numpy())
    assert_eq(xx.numpy(), yy.detach().numpy())
    assert_eq(x.numpy(), y.detach().numpy())
    # backward
    yyy.backward()
    xxx.backward()
    assert_eq(xxx.grad, yyy.grad)
    assert_eq(xx.grad, yy.grad)
    assert_eq(x.grad, y.grad)
    # axis = None
    x, y = Tensor(a), tensor(a, requires_grad=True)
    # forward
    xx = x.max()
    yy = y.max()
    yy.retain_grad()
    assert_eq(xx.numpy(), yy.detach().numpy())
    assert_eq(x.numpy(), y.detach().numpy())
    # backward
    yy.backward()
    xx.backward()
    assert_eq(xx.grad, yy.grad)
    assert_eq(x.grad, y.grad)


def test_exp():
    a = np.random.randn(*shape)
    x, y = Tensor(a), tensor(a, requires_grad=True)
    # forward
    xx = x.exp()
    yy = y.exp()
    yy.retain_grad()
    xxx, yyy = xx.sum(), yy.sum()
    yyy.retain_grad()
    assert_eq(xxx.numpy(), yyy.detach().numpy())
    assert_eq(xx.numpy(), yy.detach().numpy())
    assert_eq(x.numpy(), y.detach().numpy())
    # backward
    yyy.backward()
    xxx.backward()
    assert_eq(xxx.grad, yyy.grad)
    assert_eq(xx.grad, yy.grad)
    assert_eq(x.grad, y.grad)


def test_sub():
    a = np.random.randn(*shape)
    b = np.random.randn(*shape)
    x, y = Tensor(a), tensor(a, requires_grad=True)
    # forward
    xx = x.sub(Tensor(b))
    yy = y.sub(tensor(b))
    yy.retain_grad()
    xxx, yyy = xx.sum(), yy.sum()
    yyy.retain_grad()
    assert_eq(xxx.numpy(), yyy.detach().numpy())
    assert_eq(xx.numpy(), yy.detach().numpy())
    assert_eq(x.numpy(), y.detach().numpy())
    # backward
    yyy.backward()
    xxx.backward()
    assert_eq(xxx.grad, yyy.grad)
    assert_eq(xx.grad, yy.grad)
    assert_eq(x.grad, y.grad)


def test_add():
    a = np.random.randn(*shape)
    b = np.random.randn(*shape)
    x, y = Tensor(a), tensor(a, requires_grad=True)
    # forward
    xx = x.add(Tensor(b))
    yy = y.add(tensor(b))
    yy.retain_grad()
    xxx, yyy = xx.sum(), yy.sum()
    yyy.retain_grad()
    assert_eq(xxx.numpy(), yyy.detach().numpy())
    assert_eq(xx.numpy(), yy.detach().numpy())
    assert_eq(x.numpy(), y.detach().numpy())
    # backward
    yyy.backward()
    xxx.backward()
    assert_eq(xxx.grad, yyy.grad)
    assert_eq(xx.grad, yy.grad)
    assert_eq(x.grad, y.grad)


def test_div():
    a = np.random.randn(*shape)
    b = np.random.randn(*shape)
    x, y = Tensor(a), tensor(a, requires_grad=True)
    # forward
    xx = x.div(Tensor(b))
    yy = y.div(tensor(b))
    yy.retain_grad()
    xxx, yyy = xx.sum(), yy.sum()
    yyy.retain_grad()
    assert_eq(xxx.numpy(), yyy.detach().numpy())
    assert_eq(xx.numpy(), yy.detach().numpy())
    assert_eq(x.numpy(), y.detach().numpy())
    # backward
    yyy.backward()
    xxx.backward()
    assert_eq(xxx.grad, yyy.grad)
    assert_eq(xx.grad, yy.grad)
    assert_eq(x.grad, y.grad)


def test_mul():
    a = np.random.randn(*shape)
    b = np.random.randn(*shape)
    x, y = Tensor(a), tensor(a, requires_grad=True)
    # forward
    xx = x.mul(Tensor(b))
    yy = y.mul(tensor(b))
    yy.retain_grad()
    xxx, yyy = xx.sum(), yy.sum()
    yyy.retain_grad()
    assert_eq(xxx.numpy(), yyy.detach().numpy())
    assert_eq(xx.numpy(), yy.detach().numpy())
    assert_eq(x.numpy(), y.detach().numpy())
    # backward
    yyy.backward()
    xxx.backward()
    assert_eq(xxx.grad, yyy.grad)
    assert_eq(xx.grad, yy.grad)
    assert_eq(x.grad, y.grad)


def test_softmax():
    a = np.random.randn(*shape)
    x, y = Tensor(a), tensor(a, requires_grad=True)
    # forward
    xx = x.softmax()
    yy = softmax(y, dim=-1)
    yy.retain_grad()
    assert_eq(xx.numpy(), yy.detach().numpy())
    xxx, yyy = xx.sum(), yy.sum()
    yyy.retain_grad()
    # backward
    yyy.backward()
    xxx.backward()
    assert_eq(xxx.grad, yyy.grad)
    assert_eq(xx.grad, yy.grad)
    assert_eq(x.grad, y.grad)


def test_cross_entropy():
    # TODO
    assert True
