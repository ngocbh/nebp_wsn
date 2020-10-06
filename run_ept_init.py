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
import solver_mhn_kruskal
import solver_mhn_nrk
import solver_mhn_prim
import summarization
import run

import os
from os.path import join
from random import Random
import pandas as pd
import pickle

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(WORKING_DIR, './configs/_configurations.yml')
DATA_DIR = os.path.join(WORKING_DIR, "./data/small/multi_hop")

INIT_METHODS = ['PrimRST', 'KruskalRST', 'RandWalkRST', 'CPrimRST', 'Mix_1', 'Mix_2']
INIT_METHODS_LEGEND = ['prim', 'kruskal', 'randwalk', 'cprim', 'mix_1', 'mix_2']

RERUN=False

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
    def run(filepath, out_test_dir):
        ds = []
        for i in range(len(INIT_METHODS)):

            n_hop = 30
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
                d.append((hop, percent))
            ds.append(d)

        with open(join(out_test_dir,'ept_1.data'), 'wb') as f:
            pickle.dump(ds, f)
        open(os.path.join(out_test_dir, 'done.flag'), 'a').close()
    
    def plot(out_test_dir):
        ds = None
        with open(join(out_test_dir, 'ept_1.data'), 'rb') as f:
            ds = pickle.load(f)
        plt.figure()
        ax = plt.figure().gca()
        for d in ds:
            h = [e[0] for e in d]
            p = [e[1] for e in d]
            plt.plot(h, p)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel("feasible ratio")
        plt.xlabel("H")
        plt.title("feasible ratio comparison")
        plt.legend(INIT_METHODS_LEGEND)
        out_filepath = join(out_test_dir, 'feasible_ratio.png')
        plt.savefig(out_filepath)
        plt.close('all')

    out_dir = 'results/ept_init/ept_1'

    test_path = './data/ept_init'
    for testname in os.listdir(test_path):
        if 'dem' not in testname or (testnames is not None and testname not in testnames):
            continue
        basename = os.path.splitext(testname)[0]
        out_test_dir = join(out_dir, basename)
        os.makedirs(out_test_dir, exist_ok=True)
        filepath = join(test_path, testname)

        if not os.path.isfile(os.path.join(out_test_dir, 'done.flag')) or RERUN:
            run(filepath, out_test_dir)

        plot(out_test_dir)
        
        
    

def run_ept_2(testnames=None):
    out_dir = 'results/ept_init/ept_2'
    test_path = './data/ept_init'

    def run(filepath, out_test_dir, max_hop1, max_hop2):
        data_1 = []
        data_2 = []
        for i in range(len(INIT_METHODS)):
            n_seed = 30
            pop_size = 100
            x1, x2 = [], []
            for seed in range(1, n_seed+1):
                sol1 = init_population(filepath, INIT_METHODS[i], pop_size, max_hop1, seed)
                x1.extend([e[0] for e in sol1 if e[0] != float('inf')])
                sol2 = init_population(filepath, INIT_METHODS[i], pop_size, max_hop2, seed)
                x2.extend([e[0] for e in sol2 if e[0] != float('inf')])
            data_1.append(x1)
            data_2.append(x2)

        with open(join(out_test_dir,'ept_2.data'), 'wb') as f:
            pickle.dump((data_1, data_2), f)
        open(os.path.join(out_test_dir, 'done.flag'), 'a').close()

    def plot(out_test_dir, max_hop1, max_hop2, outname):
        
        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)

        data_1, data_2 = None, None
        with open(join(out_test_dir, 'ept_2.data'), 'rb') as f:
            data_1, data_2 = pickle.load(f)

        plt.figure()
        ax = plt.axes()

        bp = plt.boxplot(data_1, positions=[i for i in range(1, len(data_1)*3+1, 3)], widths=0.6, notch=False)
        set_box_color(bp, 'blue')
        bp = plt.boxplot(data_2, positions=[i for i in range(2, len(data_2)*3+2, 3)], widths=0.6, notch=False)
        set_box_color(bp, 'red')
        
        ax.set_xticklabels(INIT_METHODS_LEGEND)
        ax.set_xticks([i+0.5 for i in range(1, len(INIT_METHODS)*3+1, 3)])

        plt.title(outname)

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c='blue', label=f'H = {max_hop1}')
        plt.plot([], c='red', label=f'H = {max_hop2}')
        plt.legend()

        out_filepath = join(out_test_dir, outname)
        plt.savefig(out_filepath)
        plt.close('all')

    for testname in os.listdir(test_path):
        if 'dem' not in testname or (testnames is not None and testname not in testnames):
            continue
        basename = os.path.splitext(testname)[0]
        out_test_dir = join(out_dir, basename)
        os.makedirs(out_test_dir, exist_ok=True)
        filepath = join(test_path, testname)
        max_hop1, max_hop2 = 10, 20
        outname = 'relays\' distribution'
        if not os.path.isfile(os.path.join(out_test_dir, 'done.flag')) or RERUN:
            run(filepath, out_test_dir, max_hop1, max_hop2)
        plot(out_test_dir, max_hop1, max_hop2, outname)


def run_ept_3():
    def run_solver(solver, model):
        config = load_config(CONFIG_FILE, model)
        out_dir = 'results/ept_init/ept_3'

        model_dict = {}
        test_path = './data/ept_init'
        for i in range(len(INIT_METHODS)):
            smodel = '{}.{}'.format(model, i) 
            out_model_dir = os.path.join(out_dir, smodel)
            os.makedirs(out_model_dir, exist_ok=True)
            config['encoding']['init_method'] = INIT_METHODS[i]
            config['models']['gens'] = 0
            config['algorithm']['pop_size'] = 20
            print(config)
            model_dict[smodel] = smodel
            run.run_solver(solver, smodel, test_path, output_dir=out_model_dir, save_history=False, config=config, seed=42)

        summarization.summarize_model(model_dict, working_dir=out_dir, cname=f'sum-{model}')

    run_solver(solver_mhn_gprim, '1.2.5.0')
    run_solver(solver_mhn_kruskal, '1.2.2.0')
    run_solver(solver_mhn_nrk, '1.2.1.0')
    run_solver(solver_mhn_prim, '1.2.4.0')


if __name__ == '__main__':
    run_ept_1()
    # run_ept_2()
    # run_ept_3()
