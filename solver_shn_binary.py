"""
File: solver_shn_binary.py
Created by ngocjr7 on 2020-08-24 15:22
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
from problems import SingleHopProblem
from networks import SingleHopNetwork
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import sys
import yaml

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


def relay_energy(x, d):
    return wc.k_bit * (x * (wc.e_elec + wc.e_da) + wc.e_mp * d ** 4)

def sensor_energy(d):
    return wc.k_bit * (wc.e_elec + wc.e_fs * d ** 2)

def max_childs(max_energy, d):
    return int( (max_energy - wc.k_bit * wc.e_mp * d ** 4 ) / (wc.k_bit * wc.e_elec + wc.k_bit * wc.e_da) )

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

    for i in range(1, n + m + 1):
        if i <= n:
            d = distance(problem._points[i], problem._points[0])
            capacity[i] = max_childs(max_energy, d) if relay_masks[i-1] == 1 else 0
            capacity[i] = max(capacity[i], 0)
        else:
            capacity[i] = 1

    potential_edges = [set() for _ in range(n + m + 1)]
    for i in range(1, n + 1):
        for j in range(n + 1, n + m + 1):
            d = distance(problem._points[i], problem._points[j])
            if d <= 2 * problem._radius and sensor_energy(d) <= max_energy and capacity[i] != 0:
                potential_edges[i].add(j)
                potential_edges[j].add(i)

    for i in range(n + 1, n + m + 1):
        if len(potential_edges[i]) == 0:
            return False, float('inf'), [], []
    
    time = 0
    count = 0
    visited = [0] * (n + m + 1)
    assigned = [set() for _ in range(n + m + 1)]

    for i in range(n+1, n+m+1):
        time += 1
        if find_augmenting_path(i, potential_edges, capacity, assigned, visited, time):
            count += 1
        
    if count < m:
        return False, float('inf'), [], []
    
    energy = -float('inf')
    used_relays = 0
    for i in range(1, n+1):
        num_childs[i] = len(assigned[i])
        if num_childs[i] > 0:
            used_relays += 1
            parent[i] = 0
            d = distance(problem._points[0], problem._points[i])
            energy = max(energy, relay_energy(num_childs[i], d))
        for e in assigned[i]:
            parent[e] = i
    num_childs[0] = used_relays + m

    for i in range(n+1, n+m+1):
        d = distance(problem._points[i], problem._points[parent[i]])
        energy = max(energy, sensor_energy(d))

    return True, energy, parent, num_childs

def calc_max_energy_consumption(problem, relay_masks):
    # print(problem._edges)
    # print(relay_masks)
    low, high = 0, relay_energy(problem._num_of_sensors, problem._radius * 2) 
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

def solve(filename, output_dir=None, model='0.0.0.0'):
    start_time = time.time()
    config = load_config(CONFIG_FILE, model)
    check_config(config, filename, model)
    output_dir = output_dir or gen_output_dir(filename, model)

    basename, _ = os.path.splitext(os.path.basename(filename))
    os.makedirs(os.path.join(
        WORKING_DIR, '{}/{}'.format(output_dir, basename)), exist_ok=True)
    # print(basename)

    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    problem = SingleHopProblem(inp)

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
                          random_state=42)

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
        
    # indv_temp.init(random_state=45)
    # change_indv(indv_temp)
    # print(indv_temp.num_used_relays)
    # print(objective1(indv_temp))
    # print(objective2(indv_temp))
    # return

    history = engine.run(generations=config['models']['gens'])

    pareto_front = engine.get_pareto_front()
    solutions = engine.get_all_solutions()

    out_dir = os.path.join(WORKING_DIR,  f'{output_dir}/{basename}')
    end_time = time.time()

    history.dump(os.path.join(out_dir, 'history.json'))

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
