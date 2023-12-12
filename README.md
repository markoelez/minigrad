# minigrad

Simple vector-valued autodiff library.

--------------------------------------------------------------------

![Unit Tests](https://github.com/markoelez/minigrad/workflows/Unit%20Tests/badge.svg)

Inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd) and [pytorch](https://github.com/pytorch/pytorch)


## Features

### Nueral Networks

See `eval/` for concrete implementations.

Example:

```python
class NN:
    def __init__(self):
        self.l1: Tensor = Tensor.uniform(4, 10)
        self.l2: Tensor = Tensor.uniform(10, 3)

    def forward(self, x):
        x = x.dot(self.l1)
        x = x.relu()
        x = x.dot(self.l2)
        x = x.softmax()
        return x

    def __call__(self, x):
        return self.forward(x)

# input data
X_train, Y_train, X_test, Y_test = prepare(dataset)

model = NN()
optimizer = SGD(params=[model.l1, model.l2], lr=0.01)

for _ in (t := trange(10)):

    # initialize tensors
    x, y = Tensor(X_train), Tensor(Y_train)

    # forward pass
    out = model(x)

    # compute loss
    loss = out.cross_entropy(y)

    # reset gradients
    optimizer.zero_grad()

    # backward pass
    loss.backward()

    # adjust weights
    optimizer.step()

    # eval
    cat = np.argmax(out.numpy(), axis=-1)
    accuracy = (cat == np.argmax(y.numpy(), axis=-1)).mean()
    print(loss, accuracy)
```

Run with `DEBUG=1` to visualize the resulting computational graph, tensor operations, and gradient shapes:

Example:
```
********************************************************************************
[OP=<class 'minigrad.ops.CrossEntropy'>]
---------------
input grad:
---------------
()
---------------
output grads:
---------------
(50, 3)
()
********************************************************************************
[OP=<class 'minigrad.ops.Div'>]
---------------
input grad:
---------------
(50, 3)
---------------
output grads:
---------------
(50, 3)
(50, 3)
********************************************************************************
[OP=<class 'minigrad.ops.Expand'>]
---------------
input grad:
---------------
(50, 3)
---------------
output grads:
---------------
(50, 1)
********************************************************************************
[OP=<class 'minigrad.ops.Sum'>]
---------------
input grad:
---------------
(50, 1)
---------------
output grads:
---------------
(50, 3)
********************************************************************************
[OP=<class 'minigrad.ops.Exp'>]
---------------
input grad:
---------------
(50, 3)
---------------
output grads:
---------------
(50, 3)
********************************************************************************
```

Run with `DEBUG=2` to visualize the above in addition to computed gradients.