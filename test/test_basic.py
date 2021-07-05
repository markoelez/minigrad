import torch
import numpy as np
from minigrad.tensor import Tensor
from minigrad.function import Mul, Add

def test_grads():
    # minigrad
    a = Tensor(2)
    b = Tensor(3)
    c = Tensor(4)
    d = Tensor(5)

    x = Mul.apply(a, b)
    y = Mul.apply(c, d)

    z = Add.apply(x, y)

    z.backward()
    
    # pytorch
    aa = torch.tensor(2., requires_grad=True)
    bb = torch.tensor(3., requires_grad=True)
    cc = torch.tensor(4., requires_grad=True)
    dd = torch.tensor(5., requires_grad=True)
    
    xx = aa * bb
    yy = cc * dd

    zz = xx + yy

    zz.backward()

    # assert
    assert aa.detach().numpy() == a.data
    assert bb.detach().numpy() == b.data
    assert cc.detach().numpy() == c.data
    assert dd.detach().numpy() == d.data

    assert xx.detach().numpy() == x.data
    assert yy.detach().numpy() == y.data

    assert zz.detach().numpy() == z.data

    assert aa.grad == a.grad
    assert bb.grad == b.grad
    assert cc.grad == c.grad
    assert dd.grad == d.grad
