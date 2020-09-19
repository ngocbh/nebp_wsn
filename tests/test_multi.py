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
    parent = [-1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, -1, -1, 0, -1, 0, -1, 0, -1, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1, 0, -1, 0, -1, 0, 62, 70, 44, 20, 4, 61, 7, 10, 31, 15, 49, 73, 40, 5, 10, 17, 38, 63, 40, 6, 78, 9, 24, 61, 36, 60, 13, 11, 24, 44, 74, 74, 29, 56, 56, 28, 6, 14, 22, 69, 49, 107, 94, 112, 109, 94, 110, 92, 110, 49, 103, 81, 92, 100, 92, 113, 101, 93, 116, 110, 110, 104, 92, 90, 110, 113, 90, 82, 87, 102, 117, 92, 88, 51, 95, 84, 98, 97, 93, 105]
    num_childs = [100, 0, 0, 0, 1, 1, 3, 1, 0, 2, 2, 1, 0, 1, 4, 1, 0, 5, 0, 0, 4, 0, 1, 0, 4, 0, 0, 0, 1, 2, 0, 42, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 41, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 0, 3, 0, 0, 18, 1, 0, 2, 0, 0, 2, 3, 0, 19, 0, 17, 4, 2, 1, 0, 1, 2, 0, 3, 2, 14, 1, 15, 1, 0, 2, 0, 1, 13, 0, 3, 2, 0, 0, 1, 1, 0, 0, 0]
    chromosome = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0, 32, 0, 33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 40, 60, 66, 89, 110, 95, 115, 94, 100, 90, 104, 20, 44, 36, 65, 93, 119, 10, 48, 85, 109, 52, 73, 86, 94, 9, 62, 102, 110, 31, 49, 101, 110, 22, 79, 97, 118, 38, 57, 17, 56, 100, 110, 99, 116, 106, 113, 98, 117, 14, 78, 81, 92, 92, 93, 92, 103, 29, 73, 84, 112, 56, 75, 49, 81, 84, 116, 44, 70, 11, 68, 102, 104, 88, 113, 96, 113, 71, 74, 69, 80, 42, 70, 87, 110, 90, 107, 46, 61, 43, 44, 49, 90, 49, 51, 51, 114, 58, 63, 88, 92, 7, 47, 105, 120, 6, 77, 28, 76, 93, 98, 91, 103, 61, 78, 82, 107, 24, 69, 24, 63, 92, 112, 83, 94, 111, 117, 92, 95, 15, 50, 97, 101, 61, 64, 40, 53, 41, 62, 4, 45, 6, 60, 5, 54, 40, 59, 82, 108, 72, 74, 13, 67, 10, 55, 105, 110, 56, 74, 87, 109]
    # test('../data/small/multi_hop/ga-dem1_r25_1_40.json', parent, num_childs)
