"""
File: solver_mhn_prufer.py
Created by ngocjr7 on 2020-10-04 20:23
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

from geneticpython.engines import NSGAIIEngine
from geneticpython import Population
from geneticpython.core.operators import TournamentSelection
from geneticpython.tools import visualize_fronts, save_history_as_gif
from geneticpython.models.tree import PruferCode, KruskalTree, Tree
from geneticpython.core.operators import UniformCrossover, SwapMutation
from geneticpython.core.individual import IntChromosome
from geneticpython.utils.validation import check_random_state

from initalization import initialize_pop
from utils.configurations import *
from utils import WusnInput, energy_consumption
from utils import save_results
from problems import MultiHopProblem
from networks import MultiHopNetwork
from networks import WusnKruskalNetwork

from random import Random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import random
import json
import time
import yaml

import sys
import os
import copy
import numpy as np

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(WORKING_DIR, './configs/_configurations.yml')

def check_config(config, filename, model):
    if config['data']['name'] not in filename:
        raise ValueError('Model {} is used for {}, file {} is not'.format(model, config['data'], filename))
    if config['encoding']['name'] != 'prufer':
        raise ValueError('encoding {} != {}'.format(config['encoding']['name'], 'prufer'))
    if config['algorithm']['name'] != 'nsgaii':
        raise ValueError('algorithm {} != {}'.format(config['algorithm']['name'], 'nsgaii'))


class MyPruferCode(PruferCode):
    """PruferCode.
        the implementation follows tutorial: https://cp-algorithms.com/graph/pruefer_code.html
    """

    def __init__(self, number_of_vertices: int, chromosome: IntChromosome = None, solution: Tree = None, 
                 potential_edges = None, potential_edges_set = None):
        self.potential_edges = potential_edges
        self.potential_edges_set = potential_edges_set
        super(MyPruferCode, self).__init__(number_of_vertices, chromosome, solution)

    def clone(self):
        number_of_vertices = self.number_of_vertices
        solution = self.solution.clone()
        chromosome = copy.deepcopy(self.chromosome)
        
        return MyPruferCode(number_of_vertices, chromosome=chromosome, 
                            solution=solution, potential_edges=self.potential_edges, 
                            potential_edges_set=self.potential_edges_set)

    def decode(self):
        """decode.
            Decode prufer code to Tree in linear time O(n)
        """
        n = self.number_of_vertices
        code = self.chromosome.genes
        degree = [1] * n
        for i in code:
            degree[i] += 1

        ptr = 0
        while degree[ptr] != 1:
            ptr += 1

        leaf = ptr
        edges = []
        for v in code:
            edges.append((leaf, v))
            degree[v] -= 1
            if degree[v] == 1 and v < ptr:
                leaf = v
            else:
                ptr += 1
                while degree[ptr] != 1:
                    ptr += 1
                leaf = ptr

        edges.append((leaf, n-1))

        valid_edges = []
        for u, v in edges:
            if (u, v) in self.potential_edges_set or (v, u) in self.potential_edges_set:
                valid_edges.append((u, v))

        self.solution.initialize()
        
        _is_valid = True
        for u, v in valid_edges:
            _is_valid &= self.solution.add_edge(u, v)

        seed = np.sum(self.chromosome.genes) 
        random_state = check_random_state(int(seed))
        order = random_state.permutation(list(range(len(self.potential_edges))))
        for i in order:
            if len(self.solution.edges) < self.solution.number_of_vertices - 1:
                u, v = self.potential_edges[i]
                self.solution.add_edge(u, v)
            else:
                break

        self.solution._is_valid = _is_valid
        self.solution.repair()
        return self.solution

def solve(filename, output_dir=None, model='0.0.0.0', config=None, save_history=True, seed=None, save=True):
    start_time = time.time()

    seed = seed or 5
    config = config or {}
    config = update_config(load_config(CONFIG_FILE, model), config)
    check_config(config, filename, model)
    output_dir = output_dir or gen_output_dir(filename, model)
    basename, _ = os.path.splitext(os.path.basename(filename))
    os.makedirs(os.path.join(
        WORKING_DIR, '{}/{}'.format(output_dir, basename)), exist_ok=True)
    print(basename)

    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    update_max_hop(config, inp)
    update_gens(config, inp)
    problem = MultiHopProblem(inp, config['data']['max_hop'])
    network = MultiHopNetwork(problem)
    node_count = problem._num_of_relays + problem._num_of_sensors + 1
    edge_count = problem._num_of_sensors
    potential_edges = problem._idx2edge
    potential_edges_set = set(problem._idx2edge)

    indv_temp = MyPruferCode(number_of_vertices=node_count, solution=network, 
                             potential_edges=potential_edges, potential_edges_set=potential_edges_set)

    # genes = [4, 0, 0, 10, 0, 10, 0, 25, 0, 0, 36, 29, 0, 0, 0, 0, 3, 7, 40, 31, 40, 13, 7, 7, 0, 7, 8, 4, 0, 9, 4, 0, 0, 0, 17, 0, 3, 0, 4] 
    # edges = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0, 32, 0, 33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 40, 0, 41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47, 0, 48, 0, 49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 56, 0, 57, 0, 58, 0, 59, 0, 60, 0, 61, 0, 62, 0, 63, 0, 64, 0, 65, 0, 66, 0, 67, 0, 68, 0, 69, 0, 70, 0, 71, 0, 72, 0, 73, 0, 74, 0, 75, 0, 76, 0, 77, 0, 78, 0, 79, 0, 80, 0, 81, 0, 82, 0, 83, 0, 84, 0, 85, 0, 86, 0, 87, 0, 88, 0, 89, 0, 90, 0, 91, 0, 92, 0, 93, 0, 94, 0, 95, 0, 96, 0, 97, 0, 98, 0, 99, 0, 100, 163, 12, 170, 60, 112, 28, 183, 86, 206, 154, 209, 222, 215, 233, 253, 300, 263, 234, 276, 248, 248, 131, 278, 247, 240, 280, 153, 154, 287, 202, 224, 264, 294, 169, 298, 110, 299, 234, 201, 268, 268, 208, 219, 300, 29, 106, 275, 296, 42, 105, 213, 219, 213, 238, 169, 190, 196, 267, 201, 260, 225, 243, 251, 287, 218, 238, 263, 285, 60, 109, 42, 127, 45, 154, 192, 224, 261, 282, 105, 274, 118, 191, 223, 234, 134, 296, 215, 222, 37, 152, 219, 224, 256, 294, 104, 232, 234, 277, 236, 276, 128, 273, 122, 146, 224, 289, 153, 259, 168, 241, 58, 180, 246, 256, 98, 119, 47, 145, 253, 280, 207, 209, 193, 198, 58, 187, 225, 237, 168, 296, 246, 284, 83, 181, 172, 189, 223, 244, 210, 290, 120, 143, 133, 173, 5, 121, 153, 251, 201, 212, 104, 213, 118, 156, 134, 282, 23, 178, 237, 296, 12, 186, 204, 233, 189, 286, 151, 155, 63, 192, 60, 175, 216, 288, 216, 247, 28, 164, 124, 127, 107, 156, 185, 270, 212, 250, 248, 259, 249, 264, 203, 287, 216, 273, 131, 216, 148, 188, 141, 152, 283, 295, 215, 216, 10, 188, 98, 134, 63, 125, 125, 177, 205, 288, 253, 292, 44, 101, 137, 173, 221, 256, 77, 173, 32, 115, 171, 245, 201, 263, 27, 198, 210, 279, 143, 240, 129, 284, 132, 205, 169, 265, 58, 117, 39, 123, 117, 197, 36, 102, 76, 116, 113, 175, 150, 167, 228, 294, 245, 253, 166, 200, 93, 166, 245, 290, 232, 242, 6, 146, 17, 156, 134, 194, 121, 126, 130, 197, 158, 294, 167, 195, 212, 226, 70, 151, 162, 282, 123, 260, 28, 161, 84, 167, 279, 297, 36, 176, 258, 285, 240, 270, 108, 135, 184, 188, 207, 235, 36, 190, 138, 187, 267, 280, 47, 140, 257, 268, 134, 262, 90, 139, 227, 240, 102, 199, 114, 291, 85, 147, 202, 231, 103, 127, 196, 272, 58, 108, 292, 293, 110, 271, 132, 211, 108, 157, 214, 217, 136, 149, 150, 174, 17, 160, 158, 281, 239, 266, 118, 266, 114, 122, 111, 176, 26, 144, 165, 188, 120, 189, 269, 295, 179, 283, 126, 149, 114, 230, 254, 283, 168, 182, 73, 142, 217, 252, 168, 220, 110, 121, 132, 229, 114, 255, 86, 179, 214, 291, 85, 159]
    # edge_list = [(edges[i], edges[i+1]) for i in range(0, len(edges), 2)]
    # network.from_edge_list(edge_list)
    # indv_temp.encode(network)
    # solution = indv_temp.decode()
    # print(solution.edges)
    # print(solution.is_valid)
    # print(solution.calc_max_energy_consumption())
    # print(solution.num_childs)
    # print(solution.num_used_relays)
    # print(solution.parent)
    # return

    population = Population(indv_temp, config['algorithm']['pop_size'])

    @population.register_initialization
    def init_population(random_state=None):
        return initialize_pop(config['encoding']['init_method'],
                              network=network, 
                              problem=problem,
                              indv_temp=indv_temp, 
                              size=population.size,
                              max_hop=problem.max_hop,
                              random_state=random_state)
    

    crossover = UniformCrossover(pc=config['encoding']['cro_prob'], pe=0.5)
    mutation = SwapMutation(pm=config['encoding']['mut_prob'])

    engine = NSGAIIEngine(population=population,
                          crossover=crossover,
                          tournament_size=config['algorithm']['tournament_size'],
                          selection_size=config['algorithm']['slt_size'],
                          mutation=mutation,
                          random_state=seed)

    @engine.minimize_objective
    def objective1(indv):
        nonlocal network
        network = indv.decode()
        if network.is_valid:
            return network.num_used_relays
        else:
            return float('inf')

    best_mr = defaultdict(lambda: float('inf'))

    @engine.minimize_objective
    def objective2(indv):
        nonlocal best_mr, network
        network = indv.decode()
        if network.is_valid:
            mec = network.calc_max_energy_consumption()
            best_mr[int(network.num_used_relays)] = min(
                mec, best_mr[int(network.num_used_relays)])
            return mec
        else:
            return float('inf')

    history = engine.run(generations=config['models']['gens'])

    pareto_front = engine.get_pareto_front()
    solutions = engine.get_all_solutions()

    end_time = time.time()

    out_dir = os.path.join(WORKING_DIR,  f'{output_dir}/{basename}')

    if save:
        history.dump(os.path.join(out_dir, 'history.json'))
        with open(os.path.join(out_dir, 'time.txt'), mode='w') as f:
            f.write(f"running time: {end_time-start_time:}")

        save_results(pareto_front, solutions, best_mr,
                     out_dir, visualization=False)

        visualize_fronts({'nsgaii': pareto_front}, show=False, save=True,
                         title=f'pareto fronts {basename}',
                         filepath=os.path.join(out_dir, 'pareto_fronts.png'),
                         objective_name=['relays', 'energy consumption'])

        # save config
        with open(os.path.join(out_dir, '_config.yml'), mode='w') as f:
            f.write(yaml.dump(config))

        with open(os.path.join(out_dir, 'r.txt'), mode='w') as f:
            f.write('{} {}'.format(problem._num_of_relays, energy_consumption(problem._num_of_sensors, 1, problem._radius * 4)))

        P = [[0, 0],[0, 0]]
        P[0][0], P[0][1] = 1, energy_consumption(problem._num_of_sensors, 1, problem._radius * 4)
        P[1][0], P[1][1] = problem._num_of_relays, energy_consumption(problem._num_of_sensors/problem._num_of_relays, 0, 0)
        with open(os.path.join(out_dir, 'P.txt'), mode='w') as f:
            f.write('{} {}\n{} {}'.format(P[0][0], P[0][1], P[1][0], P[1][1]))

    if save_history:
        save_history_as_gif(history,
                            title="NSGAII - multi-hop",
                            objective_name=['relays', 'energy'],
                            gen_filter=lambda x: (x % 5 == 0),
                            out_dir=out_dir)

    if save or save_history:
        open(os.path.join(out_dir, 'done.flag'), 'a').close()
    
    return pareto_front

if __name__ == '__main__':
    config = {'data': {'max_hop': 12},
                  'models': {'gens': 100},
          'encoding': {'init_method': 'DCPrimRST'}}
    # solve('data/_tiny/multi_hop/tiny_ga-dem2_r25_1_0.json', model = '1.7.6.0', config=config)
    solve('data/_medium/multi_hop/medium_uu-dem2_r25_1_40.json', model = '1.7.6.0', config=config)
    # solve('data/_medium/multi_hop/medium_ga-dem2_r25_1_40.json', model='1.8.6.0', config=config)
    # solve('data/_large/multi_hop/large_ga-dem3_r25_1_40.json', model='1.9.6.0', config=config)

