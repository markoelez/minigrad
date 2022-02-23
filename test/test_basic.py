import random
import math
import torch
import numpy as np
from minigrad.tensor import Tensor
from minigrad.function import Mul, Add, Dot, Sum


def get_scalar_data():
    x = random.uniform(0, 1000)
    return Tensor(x), torch.tensor(x, requires_grad=True)

def get_vector_data(r, c):
    x = np.random.rand(r, c)
    return Tensor(x), torch.tensor(x, requires_grad=True)

def is_close(a, b):
    return math.isclose(a, b, rel_tol=1e-06)

def test_scalar_grads():
    # fixtures
    a, aa = get_scalar_data()
    b, bb = get_scalar_data()
    c, cc = get_scalar_data()
    d, dd = get_scalar_data()

    # pytorch ops
    xx = aa * bb
    xx.retain_grad()
    yy = cc * dd
    yy.retain_grad()
    zz = xx + yy
    zz.retain_grad()
    zz.backward()
    
    # minigrad ops
    x = Mul.apply(a, b)
    y = Mul.apply(c, d)
    z = Add.apply(x, y)
    z.backward()

    # ensure values are equal
    assert is_close(aa.detach().item(), a.data)
    assert is_close(bb.detach().item(), b.data)
    assert is_close(cc.detach().item(), c.data)
    assert is_close(dd.detach().item(), d.data)
    assert is_close(xx.detach().item(), x.data)
    assert is_close(yy.detach().item(), y.data)
    assert is_close(zz.detach().item(), z.data)

    # ensure gradients are equal
    assert aa.grad == a.grad
    assert bb.grad == b.grad
    assert cc.grad == c.grad
    assert dd.grad == d.grad
    assert xx.grad == x.grad
    assert yy.grad == y.grad
    assert zz.grad.numpy() == z.grad

def test_vector_grads():
    # fixtures
    a, aa = get_vector_data(8, 10)
    b, bb = get_vector_data(10, 1)
    
    # pytorch ops
    cc = aa.matmul(bb)
    cc.retain_grad()
    dd = cc.sum()
    dd.retain_grad()
    dd.backward()

    # minigrad ops
    c = Dot.apply(a, b)
    d = Sum.apply(c)
    d.backward()
    
    np.testing.assert_allclose(aa.detach().numpy(), a.data)
    np.testing.assert_allclose(bb.detach().numpy(), b.data)
    np.testing.assert_allclose(cc.detach().numpy(), c.data)
    np.testing.assert_allclose(dd.detach().numpy(), d.data)
    
    np.testing.assert_allclose(aa.grad.numpy(), a.grad)
    np.testing.assert_allclose(bb.grad.numpy(), b.grad)
    np.testing.assert_allclose(cc.grad.numpy(), c.grad)
    np.testing.assert_allclose(dd.grad.numpy(), d.grad)
