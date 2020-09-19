"""
File: solver_mhn_maxflow.py
Created by ngocjr7 on 2020-09-19 14:45
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""


from geneticpython.engines import NSGAIIEngine
from geneticpython.models import BinaryIndividual
from geneticpython.core import Population
from geneticpython.core.operators import UniformCrossover, FlipBitMutation
from geneticpython.core.operators import TournamentSelection
from geneticpython.tools.visualization import save_history_as_gif, visualize_fronts

from utils import WusnInput, save_results, distance
from utils import WusnConstants as wc
from utils.configurations import load_config, gen_output_dir
from problems import MultiHopProblem
from networks import MultiHopNetwork
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import sys
import yaml
import math

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(WORKING_DIR, './configs/_configurations.yml')
EPS = 1e-7

def check_config(config, filename, model):
    if config['data']['name'] not in filename:
        raise ValueError('Model {} is used for {}, file {} is not'.format(model, config['data'], filename))
    if config['encoding']['name'] != 'binary':
        raise ValueError('encoding {} != {}'.format(config['encoding']['name'], 'binary'))
    if config['algorithm']['name'] != 'nsgaii':
        raise ValueError('algorithm {} != {}'.format(config['algorithm']['name'], 'nsgaii'))



def transmission_energy(k, d):
    d0 = math.sqrt(wc.e_fs / wc.e_mp)
    if d <= d0:
        return k * wc.e_elec + k * wc.e_fs * (d ** 2)
    else:
        return k * wc.e_elec + k * wc.e_mp * (d ** 4)

def energy_consumption(x, y, d):
    e_t = transmission_energy(wc.k_bit, d)
    e_r = x * wc.k_bit * (wc.e_elec + wc.e_da) + y * wc.k_bit * wc.e_da
    e = e_r + e_t
    return e

def max_childs(y, max_energy, d):
    return int( (max_energy - transmission_energy(wc.k_bit, d) - y * wc.k_bit * wc.e_da) \
               / (wc.k_bit * wc.e_elec + wc.k_bit * wc.e_da) )

def find_augmenting_path(u, potential_edges, capacity, assigned, visited, time):
    if visited[u] != time:
        visited[u] = time
    else:
        return False

    for v in potential_edges[u]:
        if u in assigned[v]:
            continue
        if len(assigned[v]) < capacity[v]:
            # print(f"found {v} free, add {u}")
            assigned[v].add(u)
            return True
        for up in assigned[v]:
            if find_augmenting_path(up, potential_edges, capacity, assigned, visited, time):
                # print(f"remove {up}, add {u} in {v}")
                assigned[v].remove(up)
                assigned[v].add(u)
                return True

    return False


def find_solution(problem, relay_masks, max_energy):
    n, m = problem._num_of_relays, problem._num_of_sensors
    parent = [-1] * (n + m + 1)
    num_childs = [0] * (n + m + 1)
    capacity = [0] * (n + m + 1)
    energy = float('inf')

    return True, energy, parent, num_childs

def calc_max_energy_consumption(problem, relay_masks):
    # print(problem._edges)
    # print(relay_masks)
    low, high = 0, energy_consumption(problem._num_of_sensors, 1, problem._radius * 2) 
    # found, energy, parents, num_childs = find_solution(problem, relay_masks, 0.001)
    # return 0
    while (high - low > EPS):
        mid = (high + low) / 2
        found, energy, _, _ = find_solution(problem, relay_masks, mid)
        if (found):
            high = mid
        else:
            low = mid
     
    found, energy, parents, num_childs = find_solution(problem, relay_masks, high)
    return found, energy, parents, num_childs

class WusnBinaryIndividual(BinaryIndividual):
    def __init__(self, length):
        self.parent = None
        self.num_childs = None
        self.num_used_relays = None
        self.energy = None
        super().__init__(length=length)

def solve(filename, output_dir=None, model='0.0.0.0', config=None, save_history=True, seed=None):
    start_time = time.time()

    seed = seed or 42
    config = config or load_config(CONFIG_FILE, model)
    check_config(config, filename, model)
    output_dir = output_dir or gen_output_dir(filename, model)

    basename, _ = os.path.splitext(os.path.basename(filename))
    os.makedirs(os.path.join(
        WORKING_DIR, '{}/{}'.format(output_dir, basename)), exist_ok=True)
    # print(basename)

    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    problem = MultiHopProblem(inp, config['data']['max_hop'])

    indv_temp = WusnBinaryIndividual(length=problem._num_of_relays)
    indv_temp.parent = None
    indv_temp.num_childs = None
    indv_temp.num_used_relays = None
    indv_temp.energy = None

    population = Population(indv_temp, config['algorithm']['pop_size'])
    selection = TournamentSelection(tournament_size = config['algorithm']['tournament_size'])
    crossover = UniformCrossover(pc=config['encoding']['cro_prob'], pe=config['encoding']['cro_pe'])
    mutation = FlipBitMutation(pm=config['encoding']['mut_prob']) 

    engine = NSGAIIEngine(population, selection=selection,
                          crossover=crossover,
                          mutation=mutation,
                          selection_size=config['algorithm']['slt_size'],
                          random_state=seed)

    @engine.minimize_objective
    def objective1(indv):
        nonlocal problem
        relay_masks = indv.decode()
        found, energy, parent, num_childs = calc_max_energy_consumption(problem, relay_masks)
        indv.energy = energy
        if not found:
            return float('inf')
        indv.parent = parent
        indv.num_childs = num_childs
        indv.num_used_relays = num_childs[0] - problem._num_of_sensors
        return indv.num_used_relays

    best_mr = defaultdict(lambda: float('inf'))

    @engine.minimize_objective
    def objective2(indv):
        nonlocal best_mr
        nonlocal problem

        if indv.num_used_relays is not None:
            best_mr[int(indv.num_used_relays)] = min(indv.energy, best_mr[int(indv.num_used_relays)])
            return indv.energy
        else:
            return float('inf')
        
    indv_temp.init(random_state=45)
    # print(indv_temp.num_used_relays)
    print(objective1(indv_temp))
    print(objective2(indv_temp))
    return

    history = engine.run(generations=config['models']['gens'])

    pareto_front = engine.get_pareto_front()
    solutions = engine.get_all_solutions()

    out_dir = os.path.join(WORKING_DIR,  f'{output_dir}/{basename}')
    end_time = time.time()


    with open(os.path.join(out_dir, 'time.txt'), mode='w') as f:
        f.write(f"running time : {end_time - start_time:}")

    def extract(solution):
        return solution.parent, solution.num_childs, 2

    save_results(pareto_front, solutions, best_mr,
                 out_dir, visualization=False, extract=extract)

    visualize_fronts({'nsgaii': pareto_front}, show=False, save=True,
                     title=f'pareto fronts {basename}',
                     filepath=os.path.join(out_dir, 'pareto_fronts.png'),
                     objective_name=['relays', 'energy consumption'])

    if save_history:
        history.dump(os.path.join(out_dir, 'history.json'))
        save_history_as_gif(history,
                            title="NSGAII - single-hop",
                            objective_name=['relays', 'energy'],
                            gen_filter=lambda x: (x % 1 == 0),
                            out_dir=out_dir)

    open(os.path.join(out_dir, 'done.flag'), 'a').close()
    # save config
    with open(os.path.join(out_dir, '_config.yml'), mode='w') as f:
        f.write(yaml.dump(config))
    

if __name__ == '__main__':
    solve('data/small/single_hop/ga-dem1_r25_1.json', model='1.0.3.0')
