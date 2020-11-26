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

        random_state = check_random_state(None)
        order = random_state.permutation(list(range(len(self.potential_edges))))
        for i in order:
            if len(self.solution.edges) < self.solution.number_of_vertices - 1:
                u, v = self.potential_edges[i]
                self.solution.add_edge(u, v)

        self.solution._is_valid = _is_valid
        self.solution.repair()
        return self.solution

def solve(filename, output_dir=None, model='0.0.0.0', config=None, save_history=True, seed=None):
    start_time = time.time()

    seed = seed or 1
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

    # indv_temp.update_genes([0, 0, 0, 0, 0, 3, 4])
    # solution = indv_temp.decode()
    # print(solution.edges)
    # print(solution.is_valid)
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


    history.dump(os.path.join(out_dir, 'history.json'))
    with open(os.path.join(out_dir, 'time.txt'), mode='w') as f:
        f.write(f"running time: {end_time-start_time:}")

    save_results(pareto_front, solutions, best_mr,
                 out_dir, visualization=False)

    visualize_fronts({'nsgaii': pareto_front}, show=False, save=True,
                     title=f'pareto fronts {basename}',
                     filepath=os.path.join(out_dir, 'pareto_fronts.png'),
                     objective_name=['relays', 'energy consumption'])

    if save_history:
        save_history_as_gif(history,
                            title="NSGAII - multi-hop",
                            objective_name=['relays', 'energy'],
                            gen_filter=lambda x: (x % 5 == 0),
                            out_dir=out_dir)

    # save config
    with open(os.path.join(out_dir, '_config.yml'), mode='w') as f:
        f.write(yaml.dump(config))

    with open(os.path.join(out_dir, 'r.txt'), mode='w') as f:
        f.write('{} {}'.format(problem._num_of_relays, energy_consumption(problem._num_of_sensors, 1, problem._radius * 2)))

    P = [[0, 0],[0, 0]]
    P[0][0], P[0][1] = 1, energy_consumption(problem._num_of_sensors, 1, problem._radius * 2)
    P[1][0], P[1][1] = problem._num_of_relays, energy_consumption(problem._num_of_sensors/problem._num_of_relays, 0, 0)
    with open(os.path.join(out_dir, 'P.txt'), mode='w') as f:
        f.write('{} {}\n{} {}'.format(P[0][0], P[0][1], P[1][0], P[1][1]))

    open(os.path.join(out_dir, 'done.flag'), 'a').close()

if __name__ == '__main__':
    config = {'data': {'max_hop': 12},
                  'models': {'gens': 100},
          'encoding': {'init_method': 'DCPrimRST'}}
    # solve('data/_tiny/multi_hop/tiny_uu-dem2_r25_1_0.json', model = '1.7.2.0', config=config)
    solve('data/_medium/multi_hop/medium_ga-dem2_r25_1_40.json', model='1.8.6.0', config=config)

