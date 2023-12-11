from minigrad.tensor import Tensor


class Optimizer:
    def __init__(self, params: list[Tensor], lr: float):
        self.params: list[Tensor] = params
        self.lr: Tensor = Tensor([lr])

    def zero_grad(self):
        for x in self.params: x.grad = None

    def step(self):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float = 0.001):
        super().__init__(params, lr)

    def step(self):
        for t in self.params:
            assert t.grad
            # update weights
            d = t.numpy() - t.grad * self.lr
            t.assign(d)