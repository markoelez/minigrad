import numpy as np
from typing import Any, List, Optional
from minigrad.tensor import Tensor
from functools import partialmethod


class Context:
    def __init__(self, func, *tensors):
        self.func = func
        self.children = tensors
        self.saved_data = []

    def save_for_backward(self, *data):
        self.saved_data.extend(data)

class Function:
    @classmethod
    def apply(cls, *args):
        ctx = Context(cls, *args)
        args = [t.data for t in args]
        res = Tensor(cls.forward(ctx, *args))
        res._ctx = ctx
        return res

    @staticmethod
    def forward(ctx: Any, args: Any) -> Any:
        '''Performs the associated operation.
        '''
        raise NotImplementedError()

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        '''Calculates the vector jacobian product for this operation.
        '''
        raise NotImplementedError()

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y

    @staticmethod
    def backward(ctx, grad_output):
        return 1, 1

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_data
        return y * grad_output, x * grad_output

class Dot(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x.dot(y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_data
        dx = grad_output.dot(y.T)
        dy = grad_output.T.dot(x).T
        return dx, dy

class Sum(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sum()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_data
        return grad_output * np.ones_like(x)

class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.maximum(x, 0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_data
        dx = grad_output.copy()
        dx[x < 0] = 0
        return dx

class Pow(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x ** y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_data
        return y * (x**(y-1.0)) * grad_output, (x**y) * np.log(x) * grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x - y

    @staticmethod
    def backward(ctx, grad_output):
        return 1, -1


# register all operations
setattr(Tensor, '__mul__', partialmethod(Mul.apply))
setattr(Tensor, '__add__', partialmethod(Add.apply))
setattr(Tensor, '__sub__', partialmethod(Sub.apply))
setattr(Tensor, 'dot', partialmethod(Dot.apply))
setattr(Tensor, 'matmul', partialmethod(Dot.apply))
setattr(Tensor, '__matmul__', partialmethod(Dot.apply))
setattr(Tensor, 'sum', partialmethod(Sum.apply))
setattr(Tensor, 'relu', partialmethod(ReLU.apply))
setattr(Tensor, '__pow__', partialmethod(Pow.apply))
