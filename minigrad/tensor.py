from __future__ import annotations
from minigrad.util import topological_sort, log, DEBUG
import numpy as np
from colorama import Fore
import numpy.typing as npt
from typing import Any
from math import prod  # noqa: F401 # pylint:disable=unused-import

np.random.seed(10)
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
    def apply(cls, *args, **kwargs):
        ctx = Context(cls, *args)
        res = Tensor(cls.forward(ctx, *[t.data for t in args], **kwargs))
        res._ctx = ctx
        return res

    @staticmethod
    def forward(ctx: Context, *args, **kwargs) -> Any:
        '''Performs the associated operation.
        '''
        raise NotImplementedError()

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Any:
        '''Calculates the vector jacobian product for this operation.
        '''
        raise NotImplementedError()


import minigrad.ops as ops  # noqa


class Tensor:
    def __init__(self, data: Any):
        self.data: npt.NDArray = data
        if not isinstance(data, np.ndarray):
            self.data = np.array(self.data)
        self.grad: npt.NDArray | None = None
        self._ctx: Context | None = None

    def backward(self):
        assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

        # seed root gradient
        self.grad = np.array(1)

        # iterate through reverse topological order
        for n in reversed(topological_sort(self)):

            # calculate vector jacobian products
            grads = n._ctx.func.backward(n._ctx, n.grad)
            if len(n._ctx.children) == 1:
                grads = [grads]

            if DEBUG:
                print('*' * 80)
                log(f'{Fore.LIGHTGREEN_EX}[OP={n._ctx.func}]')
                print('-' * 15)
                log(f'{Fore.LIGHTCYAN_EX}input grad:')
                print('-' * 15)
                print(n.grad)
                print('-' * 15)
                log(f'{Fore.LIGHTCYAN_EX}output grad:')
                print('-' * 15)
                for x in grads: print(x)

            # add vjp's to associated child nodes
            for c, g in zip(n._ctx.children, grads):
                c.grad = g if c.grad is None else (c.grad + g)

    def assign(self, x: Tensor) -> Tensor:
        self.data = x.data
        return self

    def numpy(self):
        return self.data

    def detach(self):
        return self.data

    def __repr__(self):
        if self.shape == (1, ):
            return f'tensor({self.data:.2f})'
        return f'tensor({np.array2string(self.data, prefix="tensor(")})'

    @property
    def shape(self):
        return self.data.shape

    # *********** initialization methods ***********
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

    # *********** unary ops ***********
    def relu(self) -> Tensor:
        return ops.ReLU().apply(self)

    def exp(self) -> Tensor:
        return ops.Exp().apply(self)

    def sum(self, axis: int | None = None, keepdim: bool = False) -> Tensor:

        # axes to reduce over
        axes = tuple(range(len(self.shape))) if axis is None else [axis]
        axes = tuple([i if i >= 0 else i + len(self.shape) for i in axes])

        out = ops.Sum.apply(self, axis=axes)

        # optionally reduce shape
        if not keepdim:
            shape = tuple([x for i, x in enumerate(self.shape) if i not in axes])
            out = ops.Reshape().apply(out, shape=shape)

        return out

    def max(self, axis: int | None = None, keepdim: bool = False) -> Tensor:

        # axes to reduce over
        axes = tuple(range(len(self.shape))) if axis is None else [axis]
        axes = tuple([i if i >= 0 else i + len(self.shape) for i in axes])

        out = ops.Max.apply(self, axis=axes)

        # optionally reduce shape
        if not keepdim:
            shape = tuple([x for i, x in enumerate(self.shape) if i not in axes])
            out = ops.Reshape().apply(out, shape=shape)

        return out

    def mean(self, axis: int | None = None) -> Tensor:
        d = Tensor(np.array([1 / self.data.size]))
        return self.sum(axis=axis) * d

    def broadcasted(self, y: Tensor):
        x: Tensor = self

        if x.shape == y.shape: return x, y

        # Pad the tensor with fewer dimensions with ones at the beginning
        d = len(x.shape) - len(y.shape)
        if d > 0:
            y = y.reshape((1,) * d + y.shape)
        elif d < 0:
            x = x.reshape((1,) * -d + x.shape)

        # Determine the broadcasted shape
        shape = tuple(max(dx, dy) for dx, dy in zip(x.shape, y.shape))

        # Expand dimensions of x and y to the broadcasted shape
        if x.shape != shape: x = x.expand(shape)
        if y.shape != shape: y = y.expand(shape)

        return x, y

    def expand(self, shape) -> Tensor:
        shape = tuple([x if x != -1 else s for s, x in zip(self.shape, shape)])
        return ops.Expand().apply(self, shape=shape)

    def reshape(self, shape) -> Tensor:
        shape = tuple([-prod(self.shape) // prod(shape) if s == -1 else (s if s is not None else self.shape[i]) for i, s in enumerate(shape)])
        return ops.Reshape().apply(self, shape=shape)

    # *********** binary ops ***********
    def mul(self, x: Tensor) -> Tensor:
        return ops.Mul().apply(self, x)

    def div(self, x: Tensor) -> Tensor:
        return ops.Div().apply(*self.broadcasted(x))

    def add(self, x: Tensor) -> Tensor:
        return ops.Add().apply(self, x)

    def sub(self, x: Tensor) -> Tensor:
        return ops.Sub().apply(*self.broadcasted(x))

    def dot(self, x: Tensor) -> Tensor:
        return ops.Dot().apply(self, x)

    # *********** magic methods ***********
    def __mul__(self, x: Tensor) -> Tensor: return self.mul(x)
    def __div__(self, x: Tensor) -> Tensor: return self.div(x)
    def __add__(self, x: Tensor) -> Tensor: return self.add(x)
    def __sub__(self, x: Tensor) -> Tensor: return self.sub(x)
    def __matmul__(self, x: Tensor) -> Tensor: return self.dot(x)

    # *********** nn ops ***********
    def dropout(self, p: float = 0.5) -> Tensor:
        raise NotImplementedError()

    def softmax(self, axis: int = -1) -> Tensor:
        x = self - self.max(axis=axis, keepdim=True)
        e = x.exp()
        s = e.sum(axis=axis, keepdim=True)
        return e.div(s)

    def cross_entropy(self, Y: npt.NDArray) -> Tensor:
        return ops.CrossEntropy().apply(self, Y)
