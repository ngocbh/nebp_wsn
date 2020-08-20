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
    problem = MultiHopProblem(inp)
    
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
    parent = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 33, 40, 34, 15, 34, 72, 34, 1, 15, 17, 72, 34, 33, 8, 33, 40, 34, 34, 8, 34, 40, 40, 34, 15, 8, 33, 8, 40, 33, 33, 26, 33, 33, 15, 8, 8, 15, 33, 72]
    num_childs = [70, 1, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 9, 8, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    test('data/small/multi_hop/ga-dem1_r25_1_0.json', parent, num_childs)
