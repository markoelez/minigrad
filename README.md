# minigrad

Simple vector-valued autodiff library.

--------------------------------------------------------------------

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

        self.params = [self.l1, self.l2]

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
optim = Adam(params=model.params, lr=0.001)

epochs = 1000
batch_size = 128

for _ in (t := trange(epochs)):

    # reset gradients
    optim.zero_grad()

    # select batch
    idx = np.random.choice(len(X_train), batch_size, replace=False)

    # initialize tensors
    x, y = Tensor(X_train[idx]), Tensor(Y_train[idx])

    # forward pass
    out = model(x)

    # compute loss
    loss = out.cross_entropy(y)

    # backward pass
    loss.backward()

    # adjust weights
    optim.step()

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
