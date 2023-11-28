import numpy as np
from typing import Any, List, Optional
from minigrad.tensor import Function


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
        return y * (x**(y - 1.0)) * grad_output, (x**y) * np.log(x) * grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x - y

    @staticmethod
    def backward(ctx, grad_output):
        return 1, -1
