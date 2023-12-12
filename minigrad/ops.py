import numpy as np
import numpy.typing as npt
from minigrad.tensor import Function, Context


class Add(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        return x + y

    @staticmethod
    def backward(ctx: Context, grad_output):
        return 1, 1


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx: Context, grad_output):
        x, y = ctx.saved_data
        return y * grad_output, x * grad_output


class Dot(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        ctx.save_for_backward(x, y)
        return x.dot(y)

    @staticmethod
    def backward(ctx: Context, grad_output):
        x, y = ctx.saved_data
        dx = grad_output.dot(y.T)
        dy = grad_output.T.dot(x).T
        return dx, dy


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, x):
        ctx.save_for_backward(x)
        return np.maximum(x, 0)

    @staticmethod
    def backward(ctx: Context, grad_output):
        x, = ctx.saved_data
        dx = grad_output.copy()
        dx[x < 0] = 0
        return dx


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        ctx.save_for_backward(x, y)
        return x ** y

    @staticmethod
    def backward(ctx: Context, grad_output):
        x, y = ctx.saved_data
        return y * (x**(y - 1.0)) * grad_output, (x**y) * np.log(x) * grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        return x - y

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output, -grad_output


class Div(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        res = x / y
        ctx.save_for_backward(x, y)
        return res

    @staticmethod
    def backward(ctx: Context, grad_output):
        x, y = ctx.saved_data
        grad_x = grad_output / y
        grad_y = -grad_output * x / (y ** 2)
        return grad_x, grad_y


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, x):
        out = np.exp(x.data)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        out, = ctx.saved_data
        return grad_output * out


class Max(Function):
    @staticmethod
    def forward(ctx: Context, x, axis=None):
        out = np.max(x, axis=axis, keepdims=True)
        ctx.save_for_backward(x, axis)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        x, axis = ctx.saved_data
        mask = (x == np.max(x, axis=axis, keepdims=True))
        out = (mask * grad_output).astype(np.float64)
        out /= np.sum(mask, axis=axis, keepdims=True)
        return out


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, x: npt.NDArray, axis: tuple[int]):
        ctx.save_for_backward(x.shape)
        return np.sum(x, axis=axis, keepdims=True)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output * np.ones(*ctx.saved_data)


class CrossEntropy(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        # Apply softmax to x
        x_max = np.max(x.data, axis=-1, keepdims=True)
        e_x = np.exp(x.data - x_max)
        softmax_x = e_x / np.sum(e_x, axis=-1, keepdims=True)

        # Save softmax output for use in backward pass
        ctx.save_for_backward(softmax_x, y.data)

        # Compute cross-entropy loss
        cross_entropy_loss = -np.sum(y.data * np.log(softmax_x + 1e-15), axis=-1)  # Adding epsilon for numerical stability
        return np.mean(cross_entropy_loss)

    @staticmethod
    def backward(ctx: Context, grad_output):
        softmax_x, y = ctx.saved_data

        grad_x = softmax_x - y
        grad_x /= y.shape[0]

        return grad_x * grad_output, 0


class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, x, shape):
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)

    @staticmethod
    def backward(ctx: Context, grad_output: npt.NDArray):
        shape, = ctx.saved_data
        return grad_output.reshape(shape)


class Expand(Function):
    @staticmethod
    def forward(ctx: Context, x, shape):
        ctx.save_for_backward(x.shape)
        return np.broadcast_to(x, shape)

    @staticmethod
    def backward(ctx: Context, grad_output: npt.NDArray):
        shape, = ctx.saved_data
        axis = tuple(i for i, (a, b) in enumerate(zip(grad_output.shape, shape)) if a != b)
        return np.sum(grad_output, axis=axis, keepdims=True)
