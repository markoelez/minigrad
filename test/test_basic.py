import random
import math
import torch
import numpy as np
from minigrad.tensor import Tensor
from minigrad.function import Mul, Add


def get_data():
    x = random.uniform(0, 1000)
    return Tensor(x), torch.tensor(x, requires_grad=True)

def is_close(a, b):
    return math.isclose(a, b, rel_tol=1e-07)

def test_scalar_grads():
    # fixtures
    a, aa = get_data()
    b, bb = get_data()
    c, cc = get_data()
    d, dd = get_data()
    
    # minigrad ops
    x = Mul.apply(a, b)
    y = Mul.apply(c, d)
    z = Add.apply(x, y)
    z.backward()
    
    # pytorch ops
    xx = aa * bb
    yy = cc * dd
    zz = xx + yy
    zz.backward()

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
