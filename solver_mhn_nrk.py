#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Filename: multi_hop_nsgaii.py
# Description:
# Created by ngocjr7 on [14-06-2020 21:22:52]
"""
from __future__ import absolute_import

from geneticpython.tools.visualization import save_history_as_gif, visualize_fronts
from geneticpython.engines import NSGAIIEngine
from geneticpython.models.tree import NetworkRandomKeys
from geneticpython import Population
from geneticpython.core.operators import TournamentSelection, SBXCrossover, PolynomialMutation
from geneticpython.core.operators import UniformCrossover, SwapMutation

from utils.configurations import *
from utils import WusnInput, energy_consumption
from utils import visualize_front, make_gif, visualize_solutions, remove_file, save_results
from problems import MultiHopProblem
from networks import MultiHopNetwork
from initalization import initialize_pop

from random import Random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import random
import json
import time
import yaml

import sys
import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(WORKING_DIR, './configs/_configurations.yml')

class MultiHopIndividual(NetworkRandomKeys):
    def __init__(self, problem: MultiHopProblem, network: MultiHopNetwork, chromosome=None, use_encode=True):
        self.problem = problem
        node_count = problem._num_of_relays + problem._num_of_sensors + 1
        
        super(MultiHopIndividual, self).__init__(
            number_of_vertices=node_count, 
            potential_edges=problem.edge_list, 
            network=network, 
            chromosome=chromosome,
            use_encode=use_encode)

    def clone(self):
        network = self.network.clone()
        chromosome = deepcopy(self.chromosome)

        ret = MultiHopIndividual(self.problem, network, chromosome=chromosome, use_encode=False) 
        ret.edge_dict = self.edge_dict
        ret.use_encode = True
        return ret

def check_config(config, filename, model):
    if config['data']['name'] not in filename:
        raise ValueError('Model {} is used for {}, file {} is not'.format(model, config['data'], filename))
    if config['encoding']['name'] != 'netkeys':
        raise ValueError('encoding {} != {}'.format(config['encoding']['name'], 'netkeys'))
    if config['algorithm']['name'] != 'nsgaii':
        raise ValueError('algorithm {} != {}'.format(config['algorithm']['name'], 'nsgaii'))

def solve(filename, output_dir=None, model='0.0.0.0', config=None, save_history=True, seed=None, save=True):
    start_time = time.time()

    seed = seed or 42
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
    indv_temp = MultiHopIndividual(problem, network)

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

    # crossover = SBXCrossover(pc=config['encoding']['cro_prob'], 
    #                          distribution_index=config['encoding']['cro_di'])
    # mutation = PolynomialMutation(pm=config['encoding']['mut_prob'], 
    #                               distribution_index=config['encoding']['mut_di'])

    crossover = UniformCrossover(pc=config['encoding']['cro_prob'], pe=0.5)
    mutation = SwapMutation(pm=config['encoding']['cro_prob'])

    engine = NSGAIIEngine(population=population,
                          crossover=crossover,
                          tournament_size=config['algorithm']['tournament_size'],
                          selection_size=config['algorithm']['slt_size'],
                          mutation=mutation,
                          random_state=seed)

    @engine.minimize_objective
    def objective1(indv):
        network = indv.decode()
        if network.is_valid:
            return network.num_used_relays
        else:
            return float('inf')

    best_mr = defaultdict(lambda: float('inf'))

    @engine.minimize_objective
    def objective2(indv):
        nonlocal best_mr
        network = indv.decode()
        if network.is_valid:
            mec = network.calc_max_energy_consumption()
            best_mr[int(network.num_used_relays)] = min(
                mec, best_mr[int(network.num_used_relays)])
            return mec
        else:
            return float('inf')

    # indv_temp.random_init(2)
    # network.from_edge_list([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 27), (0, 36), (0, 28), (0, 32), (0, 21), (0, 24), (0, 40), (0, 29), (0, 33), (0, 58), (0, 25), (0, 38), (0, 30), (0, 23), (0, 59), (0, 26), (0, 52), (0, 46), (0, 31), (0, 35), (0, 34), (1, 43), (3, 50), (8, 45), (10, 49), (16, 55), (18, 44), (19, 56), (20, 57), (27, 51), (36, 22), (21, 54), (40, 39), (33, 37), (38, 60), (30, 53), (26, 42), (52, 41), (31, 47), (34, 48)])
    # indv_temp.encode(network)
    # print(indv_temp.chromosome.genes)
    # solution = indv_temp.decode()
    # print(solution._is_valid)
    # print(solution.num_used_relays)
    # print(solution.calc_max_energy_consumption())
    # return

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
            f.write('{} {}'.format(problem._num_of_relays, energy_consumption(problem._num_of_sensors, 1, problem._radius * 2)))

        P = [[0, 0],[0, 0]]
        P[0][0], P[0][1] = 1, energy_consumption(problem._num_of_sensors, 1, problem._radius * 2)
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
    solve('data/_tiny/multi_hop/tiny_ga-dem1_r25_1_40.json', model = '1.7.1.0', config=config)
    # solve('data/_medium/multi_hop/medium_ga-dem2_r25_1_0.json', model='1.8.1.0', config=config)
    # solve('data/_large/multi_hop/large_ga-dem1_r25_1_40.json', model='1.9.1.0', config=config)
