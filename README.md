# minigrad

Simple vector-valued autodiff library.

--------------------------------------------------------------------

![Unit Tests](https://github.com/markoelez/minigrad/workflows/Unit%20Tests/badge.svg)

Inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd) and [pytorch](https://github.com/pytorch/pytorch)

### examples

```python
from minigrad.tensor import Tensor


x = Tensor.eye(8)
y = Tensor.randn(8, 12)

z = a.dot(b).sum()
z.backward()

print(x.grad)
print(y.grad)
```
