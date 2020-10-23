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
    EPrimMutation, FPrimMutation, EOPrimMutation
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
    if config['encoding']['name'] != 'mprim4':
        raise ValueError('encoding {} != {}'.format(config['encoding']['name'], 'mprim4'))
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
    

    crossover = XPrimCrossover(pc=0.7)
    # mutation1 = WusnMutation(pm=0.2, potential_edges=problem._idx2edge) 
    # mutation2 = EPrimMutation(pm=0.5, max_hop=config['data']['max_hop'])
    mutation3 = ROPrimMutation(pm=0.5, max_hop=config['data']['max_hop'])
    mutation4 = EOPrimMutation(pm=1, max_hop=config['data']['max_hop'])
    mutations = MyMutationCompact()
    a = config['models']['gens']
    mutations.add_mutation(mutation4, (2 * a // 3) * config['algorithm']['slt_size'] )
    # mutations.add_mutation(mutation2, pm=0.5)
    mutations.add_mutation(mutation3, (a - (2 * a // 3)) * config['algorithm']['slt_size'] )
    
    # indv_temp.random_init(1)
    # # print(indv_temp.chromosome)
    # sol1 = indv_temp.decode()
    # sol2 = sol1 
    # # sol2.from_edge_list([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (0, 30), (0, 31), (0, 32), (0, 33), (0, 34), (0, 35), (0, 36), (0, 37), (0, 38), (0, 39), (0, 40), (0, 41), (0, 42), (0, 43), (0, 44), (0, 45), (0, 46), (0, 47), (0, 48), (0, 49), (0, 50), (0, 51), (0, 52), (0, 53), (0, 54), (0, 55), (0, 56), (0, 57), (0, 58), (0, 59), (0, 60), (0, 61), (0, 62), (0, 63), (0, 64), (0, 65), (0, 66), (0, 67), (0, 68), (0, 69), (0, 70), (0, 71), (0, 72), (0, 73), (0, 74), (0, 75), (0, 76), (0, 77), (0, 78), (0, 79), (0, 80), (0, 81), (0, 82), (0, 83), (0, 84), (0, 85), (0, 86), (0, 87), (0, 88), (0, 89), (0, 90), (0, 91), (0, 92), (0, 93), (0, 94), (0, 95), (0, 96), (0, 97), (0, 98), (0, 99), (0, 100), (89, 179), (10, 188), (61, 200), (70, 111), (58, 178), (45, 186), (45, 145), (67, 169), (52, 129), (39, 130), (93, 199), (27, 183), (14, 158), (36, 159), (87, 190), (21, 180), (21, 108), (90, 125), (90, 191), (90, 113), (24, 161), (77, 140), (62, 119), (99, 170), (33, 117), (71, 164), (8, 167), (96, 142), (96, 120), (46, 155), (81, 107), (81, 116), (53, 192), (43, 185), (65, 174), (65, 122), (37, 166), (22, 118), (25, 123), (78, 157), (12, 137), (63, 156), (63, 110), (100, 112), (100, 134), (85, 175), (19, 181), (9, 126), (60, 124), (54, 121), (54, 184), (57, 127), (57, 128), (95, 152), (95, 139), (29, 196), (38, 195), (23, 182), (23, 101), (13, 136), (186, 104), (129, 115), (199, 172), (183, 168), (183, 194), (190, 151), (180, 141), (191, 177), (191, 146), (113, 227), (161, 148), (140, 138), (140, 143), (142, 198), (107, 160), (174, 102), (122, 149), (122, 150), (166, 144), (110, 197), (110, 163), (112, 133), (184, 208), (127, 103), (152, 193), (152, 154), (152, 135), (196, 114), (196, 106), (101, 203), (194, 215), (151, 221), (177, 201), (177, 205), (146, 105), (227, 283), (143, 187), (198, 173), (198, 204), (150, 299), (144, 214), (163, 206), (133, 207), (208, 165), (208, 131), (135, 171), (135, 250), (135, 109), (114, 212), (106, 132), (215, 268), (221, 225), (221, 228), (205, 285), (299, 202), (214, 266), (206, 253), (206, 226), (165, 147), (131, 162), (250, 238), (225, 233), (225, 269), (225, 264), (228, 244), (285, 209), (285, 222), (202, 295), (202, 252), (253, 255), (253, 261), (226, 288), (226, 234), (264, 245), (264, 296), (264, 241), (209, 210), (222, 216), (261, 230), (288, 256), (288, 220), (234, 272), (245, 236), (241, 258), (241, 280), (241, 257), (241, 237), (241, 260), (241, 239), (216, 291), (230, 224), (220, 298), (272, 275), (272, 286), (258, 235), (257, 281), (257, 270), (257, 263), (260, 267), (239, 247), (224, 242), (298, 284), (275, 246), (270, 287), (270, 259), (263, 278), (287, 294), (259, 290), (294, 273), (256, 293), (295, 219), (195, 176), (106, 189), (156, 153), (202, 282), (224, 249), (252, 265), (202, 217), (242, 276), (217, 211), (224, 300), (224, 251), (249, 218), (249, 248), (219, 240), (295, 232), (232, 243), (232, 229), (265, 289), (240, 231), (289, 279), (211, 262), (231, 254), (211, 213), (254, 274), (262, 223), (223, 292), (254, 271), (271, 297), (297, 277)])
    # random_state = check_random_state(1)
    # stable = 0
    # for i in range(1000):
    #     indv_temp.encode(sol2)

    #     sol1 = indv_temp.decode()
    #     print(sol1.is_valid)
    #     print(sol1.num_used_relays, sol1.calc_max_energy_consumption(), end=' -> ')
    #     old = sol1.calc_max_energy_consumption()
    #     # print(sol1.edges)
    #     child = mutation4.mutate(indv_temp, random_state)
    #     # print(child.chromosome)
    #     sol2 = child.decode()
    #     if sol2.is_valid:
    #         new = sol2.calc_max_energy_consumption()
    #         print(sol2.num_used_relays, sol2.calc_max_energy_consumption())
    #     else:
    #         print(float('inf'), float('inf'))
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
    print("Number of improved EPrim mutation: {}".format(EOPrimMutation.no_improved))

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
    config = {'data': {'max_hop': 6},
                  'models': {'gens': 100},
		  'encoding': {'init_method': 'DCPrimRST'}}
    solve('data/_tiny/multi_hop/tiny_uu-dem2_r25_1_0.json', model = '1.7.9.0.1', config=config)
    # solve('data/_medium/multi_hop/medium_ga-dem1_r25_1_40.json', model='1.8.9.0', config=config)

