
def topological_sort(node):
    '''Return topological ordering of non-leaf tensors.
    '''
    res = [node]
    def topo(n):
        for c in n._ctx.children:
            if c._ctx is None:
                continue
            topo(c)
            res.append(c)
    topo(node)
    return res
