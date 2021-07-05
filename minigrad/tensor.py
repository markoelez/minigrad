from minigrad.util import topological_sort


class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = None

        self._ctx = None

    def backward(self):
        # seed root gradient
        self.grad = 1

        # iterate through reverse topological order
        for n in reversed(topological_sort(self)):
            # calculate vector jacobian products
            grads = n._ctx.func.backward(n._ctx, n.grad)
            # add vjp's to associated child nodes
            for c, g in zip(n._ctx.children, grads):
                c.grad = g if c.grad is None else (c.grad + g)

    def __repr__(self):
        return f'<Tensor: data={self.data} grad={self.grad}>'
    
    @property
    def shape(self):
        return self.data.shape
