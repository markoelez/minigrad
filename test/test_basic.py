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
    x = a * b
    y = c * d
    z = x + y
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
    cc = aa @ bb
    cc.retain_grad()
    dd = cc.sum()
    dd.retain_grad()
    dd.backward()

    # minigrad ops
    c = a @ b
    d = c.sum()
    d.backward()
    
    np.testing.assert_allclose(aa.detach().numpy(), a.data)
    np.testing.assert_allclose(bb.detach().numpy(), b.data)
    np.testing.assert_allclose(cc.detach().numpy(), c.data)
    np.testing.assert_allclose(dd.detach().numpy(), d.data)
    
    np.testing.assert_allclose(aa.grad.numpy(), a.grad)
    np.testing.assert_allclose(bb.grad.numpy(), b.grad)
    np.testing.assert_allclose(cc.grad.numpy(), c.grad)
    np.testing.assert_allclose(dd.grad.numpy(), d.grad)

def test_nn_like():
    x_d = np.random.rand(100, 784)
    y_d = np.random.rand(100, 10)

    l1_d = np.random.rand(784, 128)
    l2_d = np.random.rand(128, 10)

    lr = 0.01

    def do_torch():
        l1 = torch.tensor(l1_d.copy(), requires_grad=True)
        l2 = torch.tensor(l2_d.copy(), requires_grad=True)

        x = torch.tensor(x_d.copy(), requires_grad=True)
        y = torch.tensor(y_d.copy(), requires_grad=True)
        
        x = x @ l1
        x = x.relu()
        x = x_l2 = x @ l2
        x = x * y
        x = x.mean()
        x.backward()

        loss = x.item()

        l1.data = l1.data - lr * l1.grad
        l2.data = l2.data - lr * l2.grad

        return loss, x_l2.data, l1, l2

    def do_minigrad():
        l1 = Tensor(l1_d.copy())
        l2 = Tensor(l2_d.copy())

        x = Tensor(x_d.copy())
        y = Tensor(y_d.copy())
        
        x = x @ l1
        x = x.relu()
        x = x_l2 = x @ l2
        x = x * y
        x = x.mean()
        x.backward()

        loss = x.data

        l1.data = l1.data - lr * l1.grad
        l2.data = l2.data - lr * l2.grad

        return loss, x_l2.data, l1, l2

    loss_pt, pred_pt, l1_pt, l2_pt = do_torch()
    loss_mg, pred_mg, l1_mg, l2_mg = do_minigrad()
    
    # data
    assert is_close(loss_pt, loss_mg)
    np.testing.assert_allclose(pred_pt.detach().numpy(), pred_mg.data)
    np.testing.assert_allclose(l1_pt.detach().numpy(), l1_mg.data)
    np.testing.assert_allclose(l2_pt.detach().numpy(), l2_mg.data)

    # grads
    np.testing.assert_allclose(l1_pt.grad.numpy(), l1_mg.grad)
    np.testing.assert_allclose(l2_pt.grad.numpy(), l2_mg.grad)
