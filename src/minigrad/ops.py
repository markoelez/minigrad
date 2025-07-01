import numpy as np
import numpy.typing as npt

from minigrad.util import DEBUG
from minigrad.tensor import Context, Function


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
    (x,) = ctx.saved_data
    dx = grad_output.copy()
    dx[x < 0] = 0
    return dx


class Pow(Function):
  @staticmethod
  def forward(ctx: Context, x, y):
    ctx.save_for_backward(x, y)
    return x**y

  @staticmethod
  def backward(ctx: Context, grad_output):
    x, y = ctx.saved_data
    return y * (x ** (y - 1.0)) * grad_output, (x**y) * np.log(x) * grad_output


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
    grad_y = -grad_output * x / (y**2)
    return grad_x, grad_y


class Exp(Function):
  @staticmethod
  def forward(ctx: Context, x):
    out = np.exp(x.data)
    ctx.save_for_backward(out)
    return out

  @staticmethod
  def backward(ctx: Context, grad_output):
    (out,) = ctx.saved_data
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
    mask = x == np.max(x, axis=axis, keepdims=True)
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
    (shape,) = ctx.saved_data
    return grad_output.reshape(shape)


class Expand(Function):
  @staticmethod
  def forward(ctx: Context, x, shape):
    ctx.save_for_backward(x.shape)
    return np.broadcast_to(x, shape)

  @staticmethod
  def backward(ctx: Context, grad_output: npt.NDArray):
    (shape,) = ctx.saved_data
    axis = tuple(i for i, (a, b) in enumerate(zip(grad_output.shape, shape)) if a != b)
    return np.sum(grad_output, axis=axis, keepdims=True)


class Conv2D(Function):
  @staticmethod
  def forward(ctx: Context, x, w, stride=1, groups=1):
    if isinstance(stride, int):
      stride = (stride, stride)
    cout, cin, H, W = w.shape
    ys, xs = stride
    bs, cin_, oy, ox = x.shape[0], x.shape[1], (x.shape[2] - (H - ys)) // ys, (x.shape[3] - (W - xs)) // xs

    if DEBUG:
      print(f"cin={cin}, cout={cout}, H={H}, W={W}, cin_={cin_}")
    assert cin * groups == cin_
    assert cout % groups == 0

    rcout = cout // groups
    gx = x.reshape(bs, groups, cin, x.shape[2], x.shape[3])
    tx = np.lib.stride_tricks.as_strided(
      gx,
      shape=(bs, groups, cin, oy, ox, H, W),
      strides=(gx.strides[0], gx.strides[1], gx.strides[2], gx.strides[3] * ys, gx.strides[4] * xs, gx.strides[3], gx.strides[4]),
      writeable=False,
    )
    tw = w.reshape(groups, rcout, cin, H, W)
    ctx.save_for_backward(tx, tw, x.shape, stride, groups)
    return np.einsum("igjYXyx,gkjyx -> igkYX", tx, tw).reshape((bs, cout, oy, ox))

  @staticmethod
  def backward(ctx: Context, grad_output):
    bs, _, oy, ox = grad_output.shape
    tx, tw, x_shape, stride, groups = ctx.saved_data
    _, rcout, cin, H, W = tw.shape
    ys, xs = stride
    OY, OX = x_shape[2:4]

    ggg = grad_output.reshape(bs, groups, rcout, oy, ox)
    gdw = np.einsum("igkYX,igjYXyx -> gkjyx", ggg, tx)

    gdx = np.zeros((bs, groups, cin, OY, OX), dtype=tx.dtype)
    for Y in range(grad_output.shape[2]):
      for X in range(grad_output.shape[3]):
        iY, iX = Y * ys, X * xs
        gdx[:, :, :, iY : iY + H, iX : iX + W] += np.einsum("igk,gkjyx->igjyx", ggg[:, :, :, Y, X], tw)

    return gdx.reshape((bs, groups * cin, OY, OX)), gdw.reshape((groups * rcout, cin, H, W))


def stack_for_pool(x, py, px):
  my, mx = (x.shape[2] // py) * py, (x.shape[3] // px) * px
  stack = []
  xup = x[:, :, :my, :mx]
  for Y in range(py):
    for X in range(px):
      stack.append(xup[:, :, Y::py, X::px][None])
  return np.concatenate(stack, axis=0)


def unstack_for_pool(fxn, s, py, px):
  my, mx = (s[2] // py) * py, (s[3] // px) * px
  for Y in range(py):
    for X in range(px):
      ll = fxn(Y * px + X)
      if X == 0 and Y == 0:
        ret = np.zeros(s, dtype=ll.dtype)
      ret[:, :, Y:my:py, X:mx:px] = ll
  return ret


class MaxPool2D(Function):
  @staticmethod
  def forward(ctx: Context, x, kernel_size=(2, 2)):
    stack = stack_for_pool(x, *kernel_size)
    idxs = np.argmax(stack, axis=0)
    ctx.save_for_backward(idxs, x.shape, kernel_size)
    return np.max(stack, axis=0)

  @staticmethod
  def backward(ctx: Context, grad_output):
    idxs, s, kernel_size = ctx.saved_data
    return unstack_for_pool(lambda idx: grad_output * (idxs == idx), s, *kernel_size)
