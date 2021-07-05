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
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        '''Calculates the vector jacobian product for this operation.
        '''
        raise NotImplementedError()

class Mul(Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_outputs) -> Any:
        x, y = ctx.saved_data
        return y, x

class Add(Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor):
        return x + y

    @staticmethod
    def backward(ctx, grad_outputs) -> Any:
        return 1, 1
