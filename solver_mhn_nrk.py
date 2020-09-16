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

from utils.configurations import load_config, gen_output_dir
from utils import WusnInput
from utils import visualize_front, make_gif, visualize_solutions, remove_file, save_results
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

class MultiHopIndividual(NetworkRandomKeys):
    def __init__(self, problem: MultiHopProblem):
        self.problem = problem
        network = MultiHopNetwork(problem)
        node_count = problem._num_of_relays + problem._num_of_sensors + 1
        super(MultiHopIndividual, self).__init__(
            number_of_vertices=node_count, potential_edges=problem._idx2edge, network=network)

def check_config(config, filename, model):
    if config['data']['name'] not in filename:
        raise ValueError('Model {} is used for {}, file {} is not'.format(model, config['data'], filename))
    if config['encoding']['name'] != 'netkeys':
        raise ValueError('encoding {} != {}'.format(config['encoding']['name'], 'netkeys'))
    if config['algorithm']['name'] != 'nsgaii':
        raise ValueError('algorithm {} != {}'.format(config['algorithm']['name'], 'nsgaii'))

def solve(filename, output_dir=None, model='0.0.0.0', config=None, save_history=True):
    start_time = time.time()

    config = config or load_config(CONFIG_FILE, model)
    check_config(config, filename, model)
    output_dir = output_dir or gen_output_dir(filename, model)

    basename, _ = os.path.splitext(os.path.basename(filename))
    os.makedirs(os.path.join(
        WORKING_DIR, '{}/{}'.format(output_dir, basename)), exist_ok=True)
    print(basename)

    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    problem = MultiHopProblem(inp, config['data']['max_hop'])

    indv_temp = MultiHopIndividual(problem)

    def init_bias_genes(length, n_relays_edges, random_state=None):
        genes = np.zeros(length)
        for i in range(length):
            u = random_state.beta(config['encoding']['alpha'], config['encoding']['beta'])
            if i < n_relays_edges:
                genes[i] = 1 - u
            else:
                genes[i] = u
        return genes

    # rand = random.Random()
    # for i in range(10000):
    #     genes = init_bias_genes(problem._num_encoded_edges, problem.num_rl2ss_edges, rand)
    #     print(len(genes), problem._num_encoded_edges, problem.num_rl2ss_edges)
    #     indv_temp.update_genes(genes)
    #     network = indv_temp.decode()
    #     print(network.num_used_relays, network.max_depth)
    #     print(network.calc_max_energy_consumption())
    #     print(i, network.is_valid)
    #     if network.is_valid:
    #         break
    # return

    population = Population(indv_temp, config['algorithm']['pop_size'])

    # @population.register_initialization
    # def init_population(rand: Random = Random()):
    #     print("Initializing population")
    #     ret = []
    #     for i in range(population.size):
    #         new_indv = population.individual_temp.clone()
    #         for j in range(100):
    #             genes = init_bias_genes(
    #                 problem._num_encoded_edges, problem.num_rl2ss_edges, rand)
    #             new_indv.update_genes(genes=genes)
    #             network = new_indv.decode()
    #             if network.is_valid:
    #                 print("init sucessfullly indv number {} in {} loops".format(i, j))
    #                 break
    #         ret.append(new_indv)
    #     return ret

    selection = TournamentSelection(tournament_size=config['algorithm']['tournament_size'])
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

    end_time = time.time()

    out_dir = os.path.join(WORKING_DIR,  f'{output_dir}/{basename}')


    with open(os.path.join(out_dir, 'time.txt'), mode='w') as f:
        f.write(f"running time: {end_time-start_time:}")

    save_results(pareto_front, solutions, best_mr,
                 out_dir, visualization=False)

    visualize_fronts({'nsgaii': pareto_front}, show=False, save=True,
                     title=f'pareto fronts {basename}',
                     filepath=os.path.join(out_dir, 'pareto_fronts.png'),
                     objective_name=['relays', 'energy consumption'])

    if save_history:
        history.dump(os.path.join(out_dir, 'history.json'))
        save_history_as_gif(history,
                            title="NSGAII - multi-hop",
                            objective_name=['relays', 'energy'],
                            gen_filter=lambda x: (x % 5 == 0),
                            out_dir=out_dir)

    # save config
    with open(os.path.join(out_dir, '_config.yml'), mode='w') as f:
        f.write(yaml.dump(config))

    open(os.path.join(out_dir, 'done.flag'), 'a').close()


if __name__ == '__main__':
    solve('data/small/multi_hop/no-dem7_r50_1_0.json', model = '1.0.1.0')
