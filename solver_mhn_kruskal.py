"""
File: solver_mhn_kruskal.py
Created by ngocjr7 on 2020-08-18 16:55
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

from geneticpython.engines import NSGAIIEngine
from geneticpython import Population
from geneticpython.core.operators import TournamentSelection
from geneticpython.tools import visualize_fronts, save_history_as_gif
from geneticpython.models.tree import EdgeSets
from geneticpython.core.operators import KruskalCrossover, TreeMutation

from edge_sets import WusnMutation
from utils.configurations import load_config, gen_output_dir
from utils import WusnInput
from utils import save_results
from problems import MultiHopProblem
from networks import MultiHopNetwork

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
    if config['encoding']['name'] != 'kruskal':
        raise ValueError('encoding {} != {}'.format(config['encoding']['name'], 'kruskal'))
    if config['algorithm']['name'] != 'nsgaii':
        raise ValueError('algorithm {} != {}'.format(config['algorithm']['name'], 'nsgaii'))

def solve(filename, output_dir=None, model='0.0.0.0'):
    start_time = time.time()

    config = load_config(CONFIG_FILE, model)
    check_config(config, filename, model)
    output_dir = output_dir or gen_output_dir(filename, model)

    basename, _ = os.path.splitext(os.path.basename(filename))
    os.makedirs(os.path.join(
        WORKING_DIR, '{}/{}'.format(output_dir, basename)), exist_ok=True)
    print(basename)

    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    problem = MultiHopProblem(inp, config['data']['max_hop'])
    network = MultiHopNetwork(problem)
    node_count = problem._num_of_relays + problem._num_of_sensors + 1
    edge_count = problem._num_of_sensors
    indv_temp = EdgeSets(number_of_vertices=node_count, 
                                solution=network,
                                edge_list=problem._idx2edge)

    population = Population(indv_temp, config['algorithm']['pop_size'])

    selection = TournamentSelection(tournament_size=config['algorithm']['tournament_size'])
    crossover = KruskalCrossover(pc=config['encoding']['cro_prob'])
    mutation = WusnMutation(config['encoding']['mut_prob'], potential_edges=problem._idx2edge) 

    # print(problem._idx2edge)
    # indv_temp.random_init(2)
    # print(indv_temp.chromosome.genes)
    # solution = indv_temp.decode()
    # print(solution._is_valid)
    # indv2 = indv_temp.clone()
    # indv2.random_init(3)
    # print(indv2.chromosome.genes)
    # child1, child2 = crossover.cross(indv_temp, indv2, 3)
    # print(child1.chromosome.genes)
    # child = mutation.mutate(indv_temp, 2)
    # print(child.chromosome.genes)
    # return
    engine = NSGAIIEngine(population, selection=selection,
                          crossover=crossover,
                          mutation=mutation,
                          selection_size=config['algorithm']['slt_size'],
                          random_state=42)

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

    history.dump(os.path.join(out_dir, 'history.json'))

    with open(os.path.join(out_dir, 'time.txt'), mode='w') as f:
        f.write(f"running time: {end_time-start_time:}")

    save_results(pareto_front, solutions, best_mr,
                 out_dir, visualization=False)

    visualize_fronts({'nsgaii': pareto_front}, show=False, save=True,
                     title=f'pareto fronts {basename}',
                     filepath=os.path.join(out_dir, 'pareto_fronts.png'),
                     objective_name=['relays', 'energy consumption'])

    save_history_as_gif(history,
                        title="NSGAII - multi-hop",
                        objective_name=['relays', 'energy'],
                        gen_filter=lambda x: (x % 5 == 0),
                        out_dir=out_dir)

    open(os.path.join(out_dir, 'done.flag'), 'a').close()
    # save config
    with open(os.path.join(out_dir, '_config.yml'), mode='w') as f:
        f.write(yaml.dump(config))


if __name__ == '__main__':
    solve('data/small/multi_hop/ga-dem1_r25_1_0.json', model = '1.0.2.0')

