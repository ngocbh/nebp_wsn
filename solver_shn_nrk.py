#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*  Filename : main.py
*  Description :
*  Created by ngocjr7 on [2020-06-06 20:46]
"""
from __future__ import absolute_import
from geneticpython.tools.visualization import save_history_as_gif, visualize_fronts
from geneticpython.engines import NSGAIIEngine
from geneticpython.models.tree import NetworkRandomKeys
from geneticpython import Population
from geneticpython.core.operators import TournamentSelection, SBXCrossover, PolynomialMutation

from utils import WusnInput, visualize_front, make_gif, visualize_solutions, remove_file, save_results
from utils.configurations import load_config, gen_output_dir
from problems import SingleHopProblem
from networks import SingleHopNetwork
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import sys

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(WORKING_DIR, './configs/_configurations.yml')

class SingleHopIndividual(NetworkRandomKeys):
    def __init__(self, problem: SingleHopProblem):
        self.problem = problem
        network = SingleHopNetwork(problem)
        node_count = problem._num_of_relays + problem._num_of_sensors + 1
        super(SingleHopIndividual, self).__init__(
            number_of_vertices=node_count, edge_list=problem._idx2edge, network=network)

def check_config(config, filename, model):
    if config['data']['name'] not in filename:
        raise ValueError('Model {} is used for {}, file {} is not'.format(model, config['data'], filename))
    if config['encoding']['name'] != 'netkeys':
        raise ValueError('encoding {} != {}'.format(config['encoding']['name'], 'netkeys'))
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
    problem = SingleHopProblem(inp)

    indv_temp = SingleHopIndividual(problem)
    # rand = random.Random(seed)
    # indv_temp.init(rand=rand)
    # network = indv_temp.decode()
    # print(network.num_used_relays)
    # print(network.calc_max_energy_consumption())
    # return
    population = Population(indv_temp, config['algorithm']['pop_size'])
    selection = TournamentSelection(tournament_size = config['algorithm']['tournament_size'])
    crossover = SBXCrossover(
        pc=config['encoding']['cro_prob'], distribution_index=config['encoding']['cro_di'])
    mutation = PolynomialMutation(
        pm=config['encoding']['mut_prob'], distribution_index=config['encoding']['mut_di'])

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

    out_dir = os.path.join(WORKING_DIR,  f'{output_dir}/{basename}')
    end_time = time.time()

    history.dump(os.path.join(out_dir, 'history.json'))

    with open(os.path.join(out_dir, 'time.txt'), mode='w') as f:
        f.write(f"running time : {end_time - start_time:}")

    save_results(pareto_front, solutions, best_mr,
                 out_dir, visualization=False)

    visualize_fronts({'nsgaii': pareto_front}, show=False, save=True,
                     title=f'pareto fronts {basename}',
                     filepath=os.path.join(out_dir, 'pareto_fronts.png'),
                     objective_name=['relays', 'energy consumption'])

    save_history_as_gif(history,
                        title="NSGAII - single-hop",
                        objective_name=['relays', 'energy'],
                        gen_filter=lambda x: (x % 1 == 0),
                        out_dir=out_dir)

    open(os.path.join(out_dir, 'done.flag'), 'a').close()


if __name__ == '__main__':
    solve('data/small/single_hop/ga-dem1_r25_1.json', model='0.0.0.0')
