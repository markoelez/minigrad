# minigrad

Simple vector-valued autodiff library.

--------------------------------------------------------------------

![Unit Tests](https://github.com/markoelez/minigrad/workflows/Unit%20Tests/badge.svg)

Inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd) and [pytorch](https://github.com/pytorch/pytorch)

### examples

basic:

```python
from minigrad.tensor import Tensor


x = Tensor.eye(8)
y = Tensor.randn(8, 12)

z = a.dot(b).sum()
z.backward()

print(x.grad)
print(y.grad)
```

neural network training step (kind of):

```python
from minigrad.tensor import Tensor

lr = 0.01

# training data
x = np.random.rand(100, 784)
y = np.random.rand(100, 10)

# layers
l1 = Tensor.randn(784, 128)
l2 = Tensor.randn(128, 10)

# forward pass
x = x @ l1
x = x.relu()
x = x @ l2

# calculate loss
x = x * y
x = x.mean()

# backpropogation (gradients with respect to loss)
x.backward()

# training loss
loss = x.data

# SGD
l1.data = l1.data - lr * l1.grad
l2.data = l2.data - lr * l2.grad
```
