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

    @classmethod
    def register(cls):
        '''Register a Tensor operation.
        '''
        raise NotImplementedError()

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y

    @staticmethod
    def backward(ctx, grad_output):
        return 1, 1

    @classmethod
    def register(cls):
        setattr(Tensor, 'add', partialmethod(cls.apply))
        setattr(Tensor, '__add__', partialmethod(cls.apply))

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_data
        return y * grad_output, x * grad_output

    @classmethod
    def register(cls):
        setattr(Tensor, 'mul', partialmethod(cls.apply))
        setattr(Tensor, '__mul__', partialmethod(cls.apply))

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

    @classmethod
    def register(cls):
        setattr(Tensor, 'dot', partialmethod(cls.apply))
        setattr(Tensor, 'matmul', partialmethod(cls.apply))
        setattr(Tensor, '__matmul__', partialmethod(cls.apply))

class Sum(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sum()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_data
        return grad_output * np.ones_like(x)

    @classmethod
    def register(cls):
        setattr(Tensor, 'sum', partialmethod(cls.apply))

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

    @classmethod
    def register(cls):
        setattr(Tensor, 'relu', partialmethod(cls.apply))

class Pow(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x ** y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_data
        return y * (x**(y-1.0)) * grad_output, (x**y) * np.log(x) * grad_output

    @classmethod
    def register(cls):
        setattr(Tensor, '__pow__', partialmethod(cls.apply))
        setattr(Tensor, 'pow', partialmethod(cls.apply))

class Sub(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x - y

    @staticmethod
    def backward(ctx, grad_output):
        return 1, -1

    @classmethod
    def register(cls):
        setattr(Tensor, '__sub__', partialmethod(cls.apply))
        setattr(Tensor, 'sub', partialmethod(cls.apply))


# register all operations
for op in [Add, Mul, Dot, Sub, Sum, ReLU, Pow]:
    op.register()
