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
from geneticpython.utils import check_random_state

from edge_sets import WusnMutation, MPrimCrossover, SPrimMutation, APrimMutation, MyNSGAIIEngine, MyMutationCompact, EPrimMutation, FPrimMutation
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
    if config['encoding']['name'] != 'mprim3':
        raise ValueError('encoding {} != {}'.format(config['encoding']['name'], 'mprim3'))
    if config['algorithm']['name'] != 'nsgaii':
        raise ValueError('algorithm {} != {}'.format(config['algorithm']['name'], 'nsgaii'))

def solve(filename, output_dir=None, model='0.0.0.0', config=None, save_history=True, seed=None):
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
    # mutation1 = WusnMutation(pm=0.2, potential_edges=problem._idx2edge) 
    mutation2 = EPrimMutation(pm=1, max_hop=config['data']['max_hop'])
    mutation3 = SPrimMutation(pm=0.5, max_hop=config['data']['max_hop'])
    mutation4 = FPrimMutation(pm=1, max_hop=config['data']['max_hop'])
    mutations = MyMutationCompact()
    a = config['models']['gens']
    mutations.add_mutation(mutation2, (2 * a // 3) * config['algorithm']['slt_size'] )
    # mutations.add_mutation(mutation2, pm=0.5)
    mutations.add_mutation(mutation3, (a - (2 * a // 3)) * config['algorithm']['slt_size'] )
    
    # indv_temp.random_init(1)
    # # print(indv_temp.chromosome)
    # sol1 = indv_temp.decode()
    # sol2 = sol1 
    # random_state = check_random_state(1)
    # for i in range(1000):
    #     indv_temp.encode(sol2)

    #     sol1 = indv_temp.decode()
    #     # print(sol1.is_valid)
    #     # print(sol1.num_used_relays, sol1.calc_max_energy_consumption())
    #     child = mutation4.mutate(indv_temp, random_state)
    #     # print(child.chromosome)
    #     sol2 = child.decode()
    #     if sol2.is_valid:
    #         print(sol2.num_used_relays, sol2.calc_max_energy_consumption())
    #     else:
    #         print(float('inf'), float('inf'))
    # return

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

    print("Number of improved MPrim crossover: {}".format(MPrimCrossover.no_improved))
    print("Number of improved EPrim mutation: {}".format(EPrimMutation.no_improved))

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
    config = {'data': {'max_hop': 16},
                  'models': {'gens': 100},
		  'encoding': {'init_method': 'PrimRST'}}
    # solve('data/_tiny/multi_hop/tiny_ga-dem1_r25_1_40.json', model = '1.7.8.0.1', config=config)
    solve('data/_medium/multi_hop/medium_ga-dem1_r25_1_40.json', model='1.8.8.0', config=config)

