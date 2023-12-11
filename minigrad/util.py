import os
from colorama import Style


DEBUG = os.getenv("DEBUG")


def log(*s): return print(*s, Style.RESET_ALL)


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
