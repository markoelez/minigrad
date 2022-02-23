import numpy as np
from typing import Any, List, Optional
from minigrad.tensor import Tensor


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
        res = Tensor(cls.forward(ctx, *[t.data for t in args]))
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
        return y, x

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
