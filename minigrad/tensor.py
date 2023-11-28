from __future__ import annotations
from minigrad.util import topological_sort
import numpy as np
from typing import Any

np.set_printoptions(precision=4)


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


import minigrad.ops as ops  # noqa


class Tensor:
    def __init__(self, data):
        self.data = data
        if not isinstance(data, np.ndarray):
            self.data = np.array(self.data)
        self.grad = None
        self._ctx = None

    def backward(self):
        assert self.shape == (1, ) or self.shape == ()

        # reshape scalars
        if self.shape == ():
            self.data = self.data.reshape((1, ))

        # seed root gradient
        self.grad = np.ones_like(self.data)

        # iterate through reverse topological order
        for n in reversed(topological_sort(self)):
            # calculate vector jacobian products
            grads = n._ctx.func.backward(n._ctx, n.grad)
            if len(n._ctx.children) == 1:
                grads = [grads]
            # add vjp's to associated child nodes
            for c, g in zip(n._ctx.children, grads):
                c.grad = g if c.grad is None else (c.grad + g)

    def __repr__(self):
        if self.shape == (1, ):
            return f'tensor({self.data:.2f})'
        return f'tensor({np.array2string(self.data, prefix="tensor(")})'

    @property
    def shape(self):
        if isinstance(self.data, np.ndarray):
            return self.data.shape
        return (1, )

    def div(self, y):
        return self * (y ** Tensor(-1.0))

    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim).astype(np.float32), **kwargs)

    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.randn(*shape).astype(np.float32), **kwargs)

    @classmethod
    def uniform(cls, *shape, **kwargs):
        return cls((np.random.uniform(-1., 1., size=shape) / np.sqrt(np.prod(shape))).astype(np.float32), **kwargs)

    # unary ops
    def relu(self) -> Tensor:
        return ops.ReLU().apply(self)

    def sum(self) -> Tensor:
        return ops.Sum().apply(self)

    def mean(self) -> Tensor:
        d = Tensor(np.array([1 / self.data.size]))
        return self.sum() * d

    # binary ops
    def mul(self, other: Tensor) -> Tensor:
        return ops.Mul().apply(self, other)

    def add(self, other: Tensor) -> Tensor:
        return ops.Add().apply(self, other)

    def sub(self, other: Tensor) -> Tensor:
        return ops.Sub().apply(self, other)

    def dot(self, other: Tensor) -> Tensor:
        return ops.Dot().apply(self, other)

    # magic methods
    def __mul__(self, x: Tensor) -> Tensor: return self.mul(x)
    def __add__(self, x: Tensor) -> Tensor: return self.add(x)
    def __sub__(self, x: Tensor) -> Tensor: return self.sub(x)
    def __matmul__(self, x: Tensor) -> Tensor: return self.dot(x)
