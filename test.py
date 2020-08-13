from problems import WusnProblem, SingleHopProblem
from networks import WusnNetwork
from utils import WusnInput
from utils import WusnConstants
from utils.point import distance

import os
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

def calc_num_childs(network, parent_list):
    is_valid = True
    node_count = network.n + network.m + 1
    num_childs = [0] * node_count
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

def test(filename, parent_str):
    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    problem = SingleHopProblem(inp)
    
    network = WusnNetwork(problem) 
 
    parent_list = list(map(lambda x: int(x) + 1, parent_str.split(' ')))
    print(parent_list)
    parent = [-1] * (network.n + 1) + parent_list
    for i in range(network.n+1, network.n + network.m + 1):
        parent[parent[i]] = 0;
    parent[0] = -1

    num_used_relays = 0
    for i in range(1, network.n + 1):
        num_used_relays += ( parent[i] == 0 )
    
    num_childs, is_valid = calc_num_childs(network, parent)
    
    network.parent = parent
    network.num_childs = num_childs
    print(is_valid)
    print(parent)
    print(num_childs)
    print(num_used_relays)
    energy = network.calc_max_energy_consumption()
    print(energy)

if __name__ == '__main__':
    parent_str = "20 35 8 13 32 32 19 35 35 2 8 7 8 19 6 6 5 5 20 5 12 12 15 32 2 13 13 15 2 15 20 26 26 6 12 26 19 8 7 7"
    parent_str = "36 35 19 13 35 24 2 8 26 16 24 6 6 37 9 5 25 3 36 5 7 7 15 32 2 26 19 8 29 33 20 0 0 25 9 17 29 15 1 12"
    test('data/small/single_hop/no-dem1_r25_1.json', parent_str)
