import os
import numpy as np
import numpy.typing as npt
from colorama import Style


DEBUG = int(os.getenv("DEBUG", '0'))


def log(*s): return print(*s, Style.RESET_ALL)


def one_hot_encode(labels: npt.NDArray) -> npt.NDArray:

    m = np.unique(labels)
    h = {x: i for i, x in enumerate(m)}

    a = np.zeros((len(labels), len(h)))
    for i, x in enumerate(labels):
        a[i][h[x]] = 1
    return a


def topological_sort(node):
    '''
    Return topological ordering of tensors in computational graph.
    '''
    def topo(n, res, visited):
        visited.add(n)
        if n._ctx:
            [topo(c, res, visited) for c in n._ctx.children if c not in visited]
            res.append(n)
        return res

    return topo(node, [], set())
