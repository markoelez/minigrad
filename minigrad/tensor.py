from minigrad.util import topological_sort
import numpy as np

np.set_printoptions(precision=4)


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

    def mean(self):
        d = Tensor(np.array([1 / self.data.size]))
        return self.sum() * d

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
