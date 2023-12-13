import numpy as np
from minigrad.tensor import Tensor


class Optimizer:
    def __init__(self, params: list[Tensor], lr: float):
        self.params: list[Tensor] = params
        self.lr: float = lr

    def zero_grad(self):
        for x in self.params: x.grad = None

    def step(self):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float = 0.001):
        super().__init__(params, lr)

    def step(self):
        for t in self.params:
            assert t.grad is not None
            # update weights
            t.data = t.numpy() - t.grad * self.lr


class Adam(Optimizer):
    def __init__(self, params: list[Tensor], lr: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        super().__init__(params, lr)

        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        self.m = [np.zeros_like(t.data) for t in params]
        self.v = [np.zeros_like(t.data) for t in params]
        self.t = 0

    def step(self):
        self.t += 1
        a = self.lr * (np.sqrt(1 - np.power(self.b2, self.t)) / (1 - np.power(self.b1, self.t)))
        for i, t in enumerate(self.params):
            assert t.grad is not None
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * t.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(t.grad.data)
            t.data -= a * self.m[i] / (np.sqrt(self.v[i]) + self.eps)
