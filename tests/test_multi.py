import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from problems import WusnProblem, MultiHopProblem
from networks import MultiHopNetwork
from utils import WusnInput
from utils import WusnConstants
from utils.point import distance

import os
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

def calc_num_childs(network, parent_list):
    is_valid = True
    node_count = network.n + network.m + 1
    num_childs = [0] * node_count
    # print(network.potential_edges)
    for u, v in enumerate(parent_list):
        if v != -1 and u > network.n:
            if not u in network.potential_adj[v]:
                is_valid = False
                print(u, v)
                print(network.potential_adj[u])
                print(network.potential_adj[v])
                
            num_childs[v] += 1 
    
    num_childs[0] += network.m
    return num_childs, is_valid

def test(filename, parent, num_childs):
    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    problem = MultiHopProblem(inp, 8)
    
    network = MultiHopNetwork(problem) 
    print(network.potential_adj)
    print(network.idx2edge)
    
    parent[0] = -1
    for i in range(1, network.n + 1):
        parent[i] = (num_childs[i] == 0) * (-1)

    _, is_valid = calc_num_childs(network, parent)

    network.parent = parent
    network.num_childs = num_childs
    print(is_valid)
    print(parent)
    print(num_childs)
    energy = network.calc_max_energy_consumption()
    print(energy)

if __name__ == '__main__':
    num_childs = [27, 3, 0, 3, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1]
    parent = [-1, 0, -1, 0, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, 0, -1, 0, -1, 1, 19, 40, 34, 10, 1, 30, 4, 34, 19, 14, 4, 10, 17, 14, 3, 4, 14, 1, 3]
    test('../data/_tiny/multi_hop/tiny_ga-dem2_r25_1_0.json', parent, num_childs)
