
def topological_sort(node):
    '''Return topological ordering of tensors in computational tree.
    '''
    def topo(n, res, visited):
        visited.add(n)
        if n._ctx:
            [topo(c, res, visited) for c in n._ctx.children if c not in visited]
            res.append(n)
        return res

    return topo(node, [], set())
