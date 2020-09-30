from __future__ import absolute_import

from utils.configurations import load_config, gen_output_dir
from initalization import initialize_pop
from utils import WusnInput
from rooted_networks import MultiHopNetwork
from problems import MultiHopProblem
from geneticpython.models.tree import EdgeSets, KruskalTree
from geneticpython.utils.validation import check_random_state

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import solver_mhn_gprim
import summarization
import run

import os
from os.path import join
from random import Random
import pandas as pd

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(WORKING_DIR, './configs/_configurations.yml')
DATA_DIR = os.path.join(WORKING_DIR, "./data/small/multi_hop")

INIT_METHODS = ['PrimRST', 'KruskalRST', 'RandWalkRST', 'CPrimRST', 'Mix_1', 'Mix_2']


def objective1(indv):
    network = indv.decode()
    if network.is_valid:
        return network.num_used_relays
    else:
        return float('inf')

def objective2(indv):
    network = indv.decode()
    if network.is_valid:
        mec = network.calc_max_energy_consumption()
        return mec
    else:
        return float('inf')

def init_population(filename, init_method, size, max_hop, random_state):
    print(f"initializing : init_method:{init_method}, max_hop:{max_hop}, seed:{random_state}")
    random_state = check_random_state(random_state)
    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    problem = MultiHopProblem(inp, max_hop)
    network = MultiHopNetwork(problem)
    node_count = problem._num_of_relays + problem._num_of_sensors + 1
    indv_temp = EdgeSets(number_of_vertices=node_count, 
                         solution=network,
                         potential_edges=problem._idx2edge,
                         init_method='PrimRST')

    initialized_pop = initialize_pop(init_method, network, problem, indv_temp, size, max_hop, random_state)
    solutions = list()
    for indv in initialized_pop:
        solutions.append((objective1(indv), objective2(indv)))

    return solutions


def run_ept_1(testnames=None):
    out_dir = 'results/ept_init/ept_1'

    test_path = './data/ept_init'
    for testname in os.listdir(test_path):
        if 'dem' not in testname or (testnames is not None and testname not in testnames):
            continue
        basename = os.path.splitext(testname)[0]
        out_test_dir = join(out_dir, basename)
        os.makedirs(out_test_dir, exist_ok=True)

        ds = []
        for i in range(len(INIT_METHODS)):

            filepath = join(test_path, testname)
            n_hop = 25
            n_seed = 30
            pop_size = 100
            d = []
            for hop in range(1, n_hop+1):
                sum_valid_solutions = 0 
                for seed in range(1, n_seed+1):
                    solutions = init_population(filepath, INIT_METHODS[i], pop_size, hop, seed)
                    no_valid_soutions = sum( a != float('inf') and b != float('inf')
                                           for a, b in solutions)
                    sum_valid_solutions += no_valid_soutions

                average_valid_solutions = sum_valid_solutions / n_seed
                percent = 100 * average_valid_solutions / pop_size 
                d.append((hop, average_valid_solutions))
            ds.append(d)

        plt.figure()
        ax = plt.figure().gca()
        for i, d in enumerate(ds):
            h = [e[0] for e in d]
            p = [e[1] for e in d]
            plt.plot(h, p)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel("feasible ratio")
        plt.xlabel("H")
        plt.title("feasible ratio comparison")
        plt.legend(INIT_METHODS)
        out_filepath = join(out_test_dir, 'feasible_ratio.png')
        plt.savefig(out_filepath)
        plt.close('all')
    
def run_ept_2(testnames=None):
    pass

def run_ept_3(solver, model):
    config = load_config(CONFIG_FILE, model)
    out_dir = 'results/ept_init'

    model_dict = {}
    test_path = './data/ept_init'
    for i in range(len(INIT_METHODS)):
        smodel = '{}.{}'.format(model, i) 
        out_model_dir = os.path.join(out_dir, smodel)
        config['encoding']['init_method'] = INIT_METHODS[i]
        print(config)
        model_dict[smodel] = smodel
        run.run_solver(solver, smodel, test_path, save_history=False, config=config, seed=42)

    summarization.summarize_model(model_dict, working_dir=out_dir, cname=f'sum-{model}')


if __name__ == '__main__':
    # run_ept_1()
    # run_ept_3(solver_mhn_gprim, '1.2.5.0')
    run_ept_2()
