import numpy as np
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
        return 1, -1


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


class Max(Function):
    @staticmethod
    def forward(ctx: Context, x):
        ctx.save_for_backward(x.data)
        out = np.max(x.data, axis=-1)
        if len(out.shape) == 1: return out.reshape(-1, 1)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        x = ctx.saved_data
        mask = x == np.max(x, axis=-1, keepdims=True)
        return mask * grad_output.reshape(*grad_output.shape, 1)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, x):
        res = np.exp(x.data)
        ctx.save_for_backward(res)
        return res

    @staticmethod
    def backward(ctx: Context, grad_output):
        return ctx.saved_data * grad_output


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, x, axis=None):
        ctx.save_for_backward(x.shape)
        out = np.sum(x, axis=axis)
        if len(out.shape) == 1: return out.reshape(-1, 1)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        return np.ones(*ctx.saved_data) * grad_output


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

        # Gradient of cross-entropy with respect to x
        grad_x = softmax_x - y
        grad_x /= y.shape[0]  # Average over batch size

        return grad_x * grad_output
