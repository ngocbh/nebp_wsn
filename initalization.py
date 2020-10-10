from geneticpython.utils.validation import check_random_state
from networks import WusnKruskalNetwork
from edge_sets import cprim_init 
from rooted_networks import WusnNetwork

def kruskal_rst(network, problem, indv_temp, size, max_hop, random_state):
    kruskal_solution = WusnKruskalNetwork(problem)
    ret = []
    for _ in range(size):
        kruskal_solution.random_init(random_state)
        tmp_network = network.clone()
        tmp_network.from_edge_list(kruskal_solution.edges)

        indv = indv_temp.clone()
        indv.encode(tmp_network)

        ret.append(indv)
    return ret

def random_walk_rst(network, problem, indv_temp, size, max_hop, random_state):
    network_tmp = WusnNetwork(problem)
    network_tmp.set_initialization_method('RandWalkRST')
    for i in range(1, problem._num_of_relays+1):
        network_tmp.potential_edges.append((0,i))
    network_tmp.get_potential_adj()

    ret = []
    for _ in range(size):
        network_tmp.random_init(random_state)
        tmp_network = network.clone()
        tmp_network.from_edge_list(network_tmp.edges)

        indv = indv_temp.clone()
        indv.encode(tmp_network)

        ret.append(indv)
    return ret

def prim_rst(network, problem, indv_temp, size, max_hop, random_state):
    network_tmp = WusnNetwork(problem)
    network_tmp.set_initialization_method('PrimRST')

    ret = []
    for _ in range(size):
        network_tmp.random_init(random_state)
        tmp_network = network.clone()
        tmp_network.from_edge_list(network_tmp.edges)

        indv = indv_temp.clone()
        indv.encode(tmp_network)

        ret.append(indv)
    return ret

def cprim_rst(network, problem, indv_temp, size, max_hop, random_state):
    network_tmp = WusnNetwork(problem)
    ret = []
    for _ in range(size):
        tmp_network = cprim_init(network_tmp, problem._idx2edge, max_hop, random_state)

        network_tmp2 = network.clone()
        network_tmp2.from_edge_list(tmp_network.edges)

        indv = indv_temp.clone()
        indv.encode(network_tmp2)

        ret.append(indv)
    return ret

def initialize_pop(init_method, network, problem, indv_temp, size, max_hop, random_state=None):
    random_state = check_random_state(random_state)

    if init_method == 'KruskalRST': 
        return kruskal_rst(network, problem, indv_temp, size, max_hop, random_state)
    elif init_method == 'RandWalkRST':
        return random_walk_rst(network, problem, indv_temp, size, max_hop, random_state)
    elif init_method == 'PrimRST':
        return prim_rst(network, problem, indv_temp, size, max_hop, random_state)
    elif init_method == 'CPrimRST':
        return cprim_rst(network, problem, indv_temp, size, max_hop, random_state)
    elif init_method == 'Mix_1':
        ret = []
        ret = kruskal_rst(network, problem, indv_temp, int(size/4), max_hop, random_state)
        # ret.extend(cprim_rst(network, problem, indv_temp, int(size/4), max_hop, random_state))
        ret.extend(prim_rst(network, problem, indv_temp, int(size/2), max_hop, random_state))
        ret.extend(random_walk_rst(network, problem, indv_temp, int(size/4), max_hop, random_state))
        return ret
    elif init_method == 'Mix_2':
        ret = []
        ret = kruskal_rst(network, problem, indv_temp, int(size/2), max_hop, random_state)
        ret.extend(prim_rst(network, problem, indv_temp, int(size/2), max_hop, random_state))
        return ret
        
