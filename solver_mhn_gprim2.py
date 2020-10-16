
"""
File: solver_mhn_mprim.py
Created by ngocjr7 on 2020-09-12 15:36
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

from geneticpython.engines import NSGAIIEngine
from geneticpython import Population
from geneticpython.core.operators import TournamentSelection
from geneticpython.tools import visualize_fronts, save_history_as_gif
from geneticpython.models.tree import EdgeSets, KruskalTree
from geneticpython.core.operators import PrimCrossover, TreeMutation, MutationCompact

from edge_sets import WusnMutation, MPrimCrossover, SPrimMutation, APrimMutation, MyNSGAIIEngine, MyMutationCompact
from initalization import initialize_pop
from utils.configurations import *
from utils import WusnInput, energy_consumption
from utils import save_results
from problems import MultiHopProblem
from rooted_networks import MultiHopNetwork
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

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(WORKING_DIR, './configs/_configurations.yml')

def check_config(config, filename, model):
    if config['data']['name'] not in filename:
        raise ValueError('Model {} is used for {}, file {} is not'.format(model, config['data'], filename))
    if config['encoding']['name'] != 'mprim2':
        raise ValueError('encoding {} != {}'.format(config['encoding']['name'], 'mprim2'))
    if config['algorithm']['name'] != 'nsgaii':
        raise ValueError('algorithm {} != {}'.format(config['algorithm']['name'], 'nsgaii'))

def solve(filename, output_dir=None, model='0.0.0.0', config=None, save_history=True, seed=None):
    start_time = time.time()

    seed = seed or 42
    config = config or {}
    config = update_config(load_config(CONFIG_FILE, model), config)
    check_config(config, filename, model)
    print(config)
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

    indv_temp = EdgeSets(number_of_vertices=node_count, 
                         solution=network,
                         potential_edges=problem._idx2edge,
                         init_method='PrimRST')

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
    

    crossover = MPrimCrossover(pc=0.7)
    mutation1 = WusnMutation(pm=0.2, potential_edges=problem._idx2edge) 
    # mutation2 = APrimMutation(pm=1)
    mutation3 = SPrimMutation(pm=0.5)
    mutations = MyMutationCompact()
    a = config['models']['gens']
    mutations.add_mutation(mutation1, (2 * a // 3) * config['algorithm']['slt_size'] )
    # mutations.add_mutation(mutation2, pm=0.5)
    mutations.add_mutation(mutation3, (a - (2 * a // 3)) * config['algorithm']['slt_size'] )
    # print(problem._idx2edge)
    # indv_temp.random_init(1421)
    # edge_list = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (0, 30), (0, 31), (0, 32), (0, 33), (0, 34), (0, 35), (0, 36), (0, 37), (0, 38), (0, 39), (0, 40), (21, 78), (14, 52), (25, 44), (20, 77), (18, 54), (35, 60), (3, 59), (25, 75), (3, 61), (30, 65), (24, 73), (10, 41), (27, 68), (39, 64), (1, 69), (35, 47), (30, 57), (19, 50), (26, 55), (24, 79), (36, 58), (2, 70), (14, 53), (11, 46), (19, 67), (3, 42), (20, 74), (78, 62), (2, 71), (34, 63), (41, 72), (4, 66), (39, 80), (68, 56), (18, 43), (66, 49), (34, 51), (36, 48), (11, 76), (69, 45)]
    # network.from_edge_list(edge_list)
    # indv_temp.encode(network) 
    # print(indv_temp.chromosome.genes)
    # solution = indv_temp.decode()
    # print(solution._is_valid)
    # print(solution.num_used_relays)
    # print(solution.calc_max_energy_consumption())
    # print(solution.max_depth)
    # print(solution.edges)
    # child = mutation.mutate(indv_temp, 1)
    # print(child.chromosome.genes)
    # solution2 = child.decode()
    # print(solution2._is_valid)
    # print(solution2.num_used_relays)
    # print(solution2.calc_max_energy_consumption())
    
    # indv2 = indv_temp.clone()
    # indv2.random_init(1231)
    # print(indv2.chromosome.genes)
    # solution2 = indv2.decode()
    # print(solution2._is_valid)
    # print(solution2.calc_max_energy_consumption())
    # child = crossover.cross(indv_temp, indv2, 1)
    # print(child[0].chromosome.genes)
    # solution3 = child[0].decode()
    # print(solution3._is_valid)
    # print(solution3.calc_max_energy_consumption())

    # solution4 = child[1].decode()
    # print(solution4._is_valid)
    # print(solution4.calc_max_energy_consumption())
    # return

    def crowded_comparator(p1, p2):
        if p1.nondominated_rank < p2.nondominated_rank:
            if p1.nondominated_rank >= p2.nondominated_rank - 1 and \
                    p1.crowding_distance == 0 and p2.crowding_distance != 0:
                return 1
            return -1
        elif p1.nondominated_rank > p2.nondominated_rank:
            if p1.nondominated_rank - 1 <= p2.nondominated_rank and \
                    p1.crowding_distance != 0 and p2.crowding_distance == 0:
                return -1
            return 1
        else:
            if p1.crowding_distance > p2.crowding_distance:
                return -1
            elif p1.crowding_distance < p2.crowding_distance:
                return 1
            else:
                return 0

    engine = NSGAIIEngine(population=population,
                          crossover=crossover,
                          tournament_size=config['algorithm']['tournament_size'],
                          selection_size=config['algorithm']['slt_size'],
                          mutation=mutations,
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

    history = engine.run(generations=config['models']['gens'])

    pareto_front = engine.get_pareto_front()
    solutions = engine.get_all_solutions()

    end_time = time.time()

    out_dir = os.path.join(WORKING_DIR,  f'{output_dir}/{basename}')


    print("Number of improved crossover: {}".format(MPrimCrossover.no_improved))

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

    open(os.path.join(out_dir, 'done.flag'), 'a').close()

if __name__ == '__main__':
    config = {'data': {'max_hop': 10},
                  'models': {'gens': 100},
		  'encoding': {'init_method': 'PrimRST'}}
    solve('data/_tiny/multi_hop/tiny_ga-dem1_r25_1_40.json', model = '1.7.7.0.0', 
          config=config)

