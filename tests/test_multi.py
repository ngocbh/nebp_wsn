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
        if v != -1:
            if not u in network.potential_edges[v]:
                is_valid = False
                print(u, v)
                print(network.potential_edges[u])
                print(network.potential_edges[v])
                
            num_childs[v] += 1 
    
    num_childs[0] += network.m
    return num_childs, is_valid

def test(filename, parent, num_childs):
    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    problem = MultiHopProblem(inp, 8)
    
    network = MultiHopNetwork(problem) 
    
    parent[0] = -1
    for i in range(1, network.n + 1):
        parent[i] = (num_childs[i] == 0) * (-1)

    _, is_valid = calc_num_childs(network, parent)

    network.parent = parent
    network.num_childs = num_childs
    print(is_valid)
    print(parent)
    print(num_childs)
    # print(num_used_relays)
    energy = network.calc_max_energy_consumption()
    print(energy)

if __name__ == '__main__':
    parent = [0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 20, 39, 15, 57, 31, 36, 26, 25, 36, 24, 20, 19, 38, 4, 27, 8, 18, 35, 17, 3, 39, 40, 34, 11, 1, 32, 9, 34, 54, 9, 2, 7, 14, 15, 10, 60, 30, 28, 21, 23]
    num_childs =[71, 1, 1, 2, 2, 0, 0, 1, 1, 2, 1, 1, 0, 0, 1, 2, 0, 1, 2, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 2, 1, 2, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    test('../data/small/multi_hop/no-dem7_r50_1_0.json', parent, num_childs)
