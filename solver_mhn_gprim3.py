
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

from edge_sets import WusnMutation, XPrimCrossover, ROPrimMutation, \
    APrimMutation, MyNSGAIIEngine, MyMutationCompact,\
    EPrimMutation, FPrimMutation, EOPrimMutation1, EOPrimMutation, MyEdgeSets
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

    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    update_max_hop(config, inp)
    update_gens(config, inp)
    print(basename, config)
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
   

    crossover = XPrimCrossover(pc= config['encoding']['cro_prob'])
    mutation1 = WusnMutation(pm=0.1, potential_edges=problem._idx2edge) 
    # mutation2 = EPrimMutation(pm=0.5, max_hop=config['data']['max_hop'])
    mutation3 = ROPrimMutation(pm=1, max_hop=config['data']['max_hop'])
    mutation4 = EOPrimMutation1(pm=1, max_hop=config['data']['max_hop'], backup_mutation=mutation1)

    mutations = MyMutationCompact()
    a = config['models']['gens'] * config['algorithm']['slt_size']

    # mutations.add_mutation(mutation4, (3 * a // 6))
    # # mutations.add_mutation(mutation2, pm=0.5)
    # mutations.add_mutation(mutation3, (2 * a // 6))
    # mutations.add_mutation(mutation4, (1 * a // 6))

    mutations = MutationCompact()
    mutations.add_mutation(mutation4, config['encoding']['mut_prob_a'])
    mutations.add_mutation(mutation3, config['encoding']['mut_prob_b'])

    # indv_temp.random_init(2)
    # # indv_temp.update_genes([0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 14, 36, 14, 25, 16, 37, 16, 30, 10, 29, 15, 27, 18, 35, 18, 40, 5, 26, 1, 33, 1, 31, 13, 32, 13, 38, 6, 23, 29, 39, 27, 22, 23, 34, 26, 28, 2, 21, 2, 24])
    # # print(indv_temp.chromosome)
    # sol1 = indv_temp.decode()
    # sol2 = sol1 
    # # sol2.from_edge_list([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (0, 30), (0, 31), (0, 32), (0, 33), (0, 34), (0, 35), (0, 36), (0, 37), (0, 38), (0, 39), (0, 40), (0, 41), (0, 42), (0, 43), (0, 44), (0, 45), (0, 46), (0, 47), (0, 48), (0, 49), (0, 50), (0, 51), (0, 52), (0, 53), (0, 54), (0, 55), (0, 56), (0, 57), (0, 58), (0, 59), (0, 60), (0, 61), (0, 62), (0, 63), (0, 64), (0, 65), (0, 66), (0, 67), (0, 68), (0, 69), (0, 70), (0, 71), (0, 72), (0, 73), (0, 74), (0, 75), (0, 76), (0, 77), (0, 78), (0, 79), (0, 80), (0, 81), (0, 82), (0, 83), (0, 84), (0, 85), (0, 86), (0, 87), (0, 88), (0, 89), (0, 90), (0, 91), (0, 92), (0, 93), (0, 94), (0, 95), (0, 96), (0, 97), (0, 98), (0, 99), (0, 100), (1, 178), (3, 174), (4, 146), (4, 161), (4, 104), (8, 116), (9, 129), (14, 167), (17, 179), (23, 120), (23, 107), (25, 168), (26, 139), (32, 171), (34, 114), (39, 180), (43, 196), (44, 108), (44, 131), (46, 191), (46, 156), (47, 118), (48, 183), (49, 160), (49, 181), (51, 165), (53, 192), (57, 154), (57, 143), (58, 159), (61, 152), (61, 123), (63, 148), (64, 103), (72, 124), (73, 164), (73, 190), (74, 134), (76, 113), (81, 182), (85, 158), (87, 169), (88, 185), (88, 153), (90, 175), (90, 101), (92, 122), (93, 106), (93, 137), (95, 125), (96, 111), (97, 126), (99, 117), (99, 157), (146, 147), (161, 105), (129, 155), (129, 149), (107, 266), (107, 170), (168, 121), (168, 194), (114, 115), (180, 195), (196, 142), (196, 136), (191, 173), (183, 184), (154, 135), (143, 109), (159, 199), (103, 150), (103, 201), (164, 162), (164, 140), (164, 208), (134, 166), (113, 172), (169, 151), (185, 285), (175, 110), (122, 141), (106, 189), (106, 177), (111, 188), (111, 119), (155, 128), (155, 205), (149, 212), (194, 215), (115, 299), (195, 102), (136, 145), (136, 138), (136, 130), (109, 221), (199, 204), (199, 132), (201, 217), (166, 144), (166, 214), (166, 203), (110, 163), (141, 200), (141, 198), (141, 127), (177, 282), (205, 206), (215, 268), (299, 186), (299, 209), (145, 197), (221, 225), (221, 238), (204, 207), (132, 187), (217, 242), (217, 202), (163, 222), (200, 176), (282, 252), (209, 255), (209, 253), (209, 226), (197, 112), (197, 193), (225, 269), (238, 233), (238, 228), (238, 264), (207, 283), (242, 276), (242, 232), (242, 240), (242, 218), (252, 219), (255, 249), (255, 286), (255, 230), (226, 288), (226, 234), (193, 133), (269, 278), (233, 296), (233, 258), (228, 244), (283, 227), (276, 231), (240, 243), (240, 236), (218, 223), (219, 265), (219, 250), (219, 229), (249, 213), (249, 293), (286, 261), (286, 210), (286, 211), (286, 272), (230, 224), (288, 216), (288, 220), (234, 284), (278, 281), (278, 241), (296, 267), (258, 237), (231, 279), (243, 254), (236, 257), (265, 248), (265, 300), (213, 246), (213, 292), (213, 262), (261, 256), (224, 251), (220, 298), (220, 291), (281, 247), (237, 280), (254, 290), (254, 274), (254, 271), (257, 239), (248, 295), (248, 273), (248, 289), (292, 275), (247, 245), (247, 235), (247, 263), (271, 297), (273, 294), (273, 259), (235, 260), (297, 277), (259, 287), (287, 270)])
    # random_state = check_random_state(1)
    # stable = 0
    # for i in range(1000):
    #     indv_temp.encode(sol2)

    #     print("begin" + "="*10)
    #     sol1 = indv_temp.decode()
    #     print(sol1.num_used_relays, sol1.calc_max_energy_consumption(), sol1.is_valid, end=' -> ')
    #     old = sol1.calc_max_energy_consumption()
    #     # print(sol1.edges)
    #     child = mutation4.mutate(indv_temp, random_state)
    #     # print(child.chromosome)
    #     sol2 = child.decode()

    #     new = sol2.calc_max_energy_consumption()
    #     print(sol2.is_valid, sol2.num_used_relays, sol2.calc_max_energy_consumption())

    #     print("end" + "="*10)
    #     if abs(new - old) < 1e-10:
    #         stable += 1
    #     else:
    #         stable = 0
    #     if stable > 10:
    #         indv_temp.random_init()
    #         sol2 = indv_temp.decode()
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
        return network.get_number_of_used_relays()

    best_mr = defaultdict(lambda: float('inf'))

    @engine.minimize_objective
    def objective2(indv):
        nonlocal best_mr
        network = indv.decode()
        mec = network.calc_max_energy_consumption()
        if mec != float('inf'):
            best_mr[int(network.num_used_relays)] = min(
                mec, best_mr[int(network.num_used_relays)])
        return mec

    history = engine.run(generations=config['models']['gens'])

    pareto_front = engine.get_pareto_front()
    solutions = engine.get_all_solutions()

    end_time = time.time()

    out_dir = os.path.join(WORKING_DIR,  f'{output_dir}/{basename}')

    print("Number of improved MPrim crossover: {}".format(XPrimCrossover.no_improved))
    print("Number of improved EPrim mutation: {}".format(EOPrimMutation1.no_improved))

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
    # solve('data/_tiny/multi_hop/tiny_ga-dem1_r25_1_40.json', model = '1.7.8.0', config=config)
    solve('data/_medium/multi_hop/medium_ga-dem1_r25_1_40.json', model='1.8.8.0', config=config)

