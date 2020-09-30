from __future__ import absolute_import

from geneticpython.utils.validation import check_random_state
from geneticpython.utils import rset

def cprim_init(network, potential_edges, max_hop, random_state=None):
    random_state = check_random_state(random_state)

    ret = network.clone()

    ret.initialize()

    # Set of connected nodes
    C = set()
    # eligible edges
    A = rset()
    depth = [0] * ret.number_of_vertices
    for i in range(1, ret.n+1):
        depth[i] = 1

    # Init tree
    for u in range(ret.number_of_vertices):
        if ret.parent[u] != -1:
            C.add(u)
            for v in ret.potential_adj[u]:
                if v not in C:
                    A.add((u, v))

    while len(C) < ret.number_of_vertices and len(A) > 0:
        u, v = A.random_choice(random_state)
        A.remove((u, v))
        if v not in C and max(depth[u], depth[v]) < max_hop:
            ret.add_edge(u, v)
            if ret.parent[u] == v:
                depth[u] = depth[v] + 1
            elif ret.parent[v] == u:
                depth[v] = depth[u] + 1
            C.add(v)
            for w in ret.potential_adj[v]:
                if w not in C:
                    A.add((v, w))

    if len(C) < ret.number_of_vertices:
        # print('phase 3')
        for u in C:
            for v in ret.potential_adj[u]:
                if v not in C:
                    A.add((u, v))

    while len(C) < ret.number_of_vertices:
        u, v = A.random_choice(random_state)
        A.remove((u, v))
        if v not in C:
            ret.add_edge(u, v)
            if ret.parent[u] == v:
                depth[u] = depth[v] + 1
            elif ret.parent[v] == u:
                depth[v] = depth[u] + 1
            C.add(v)
            for w in ret.potential_adj[v]:
                if w not in C:
                    A.add((v, w))

    ret.repair()
    return ret

if __name__ == '__main__':
    pass
