from minigrad.util import topological_sort
import numpy as np

np.set_printoptions(precision=4)


class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self._ctx = None

    def backward(self):
        assert self.shape == (1, )

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
        if type(self.data) == type(np.array([])):
            return self.data.shape
        return (1, )

    def mean(self):
        d = Tensor(np.array([1 / self.data.size]))
        return self.sum() * d
