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
import solver_mhn_gprim3
import solver_mhn_gprim2
import solver_mhn_kruskal
import solver_mhn_nrk
import solver_mhn_prim
import solver_mhn_prufer
import summarization
import run

import os
from os.path import join
from random import Random
import pandas as pd
import pickle
import itertools

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(WORKING_DIR, './configs/_configurations.yml')
DATA_DIR = os.path.join(WORKING_DIR, "./data/small/multi_hop")

INIT_METHODS = ['PrimRST', 'KruskalRST',
                'RandWalkRST', 'DCPrimRST']
INIT_METHODS_LEGEND = ['PrimRST', 'KruskalRST',
                       'RandWalkRST', 'DCPrimRST']

RERUN = False
TESTING = False


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
    print(
        f"initializing : init_method:{init_method}, max_hop:{max_hop}, seed:{random_state}")
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

    initialized_pop = initialize_pop(
        init_method, network, problem, indv_temp, size, max_hop, random_state)
    solutions = list()
    for indv in initialized_pop:
        solutions.append((objective1(indv), objective2(indv)))

    return solutions


def run_ept_1(testnames=None):
    def run(filepath, out_test_dir):
        print(f"Running: {filepath}")
        ds = []
        for i in range(len(INIT_METHODS)):
            n_hop = 10 if TESTING else 40
            n_hop_start = 1 if TESTING else 0
            n_seed = 1 if TESTING else 10
            pop_size = 20 if TESTING else 100
            d = []
            for hop in range(n_hop_start, n_hop+1):
                sum_valid_solutions = 0
                for seed in range(1, n_seed+1):
                    solutions = init_population(
                        filepath, INIT_METHODS[i], pop_size, hop, seed)
                    no_valid_soutions = sum(a != float('inf') and b != float('inf')
                                            for a, b in solutions)
                    sum_valid_solutions += no_valid_soutions

                average_valid_solutions = sum_valid_solutions / n_seed
                percent = 100 * average_valid_solutions / pop_size
                d.append((hop, percent))
            ds.append(d)

        with open(join(out_test_dir, 'ept_1.data'), 'wb') as f:
            pickle.dump(ds, f)
        open(os.path.join(out_test_dir, 'done_ept_1.flag'), 'a').close()

    def plot(out_test_dir, cname=''):
        plt.style.use('seaborn-white')
        plt.grid(True)
        if 'NIn' in cname:
            cname = cname.split('_')[0]
        ds = None
        with open(join(out_test_dir, 'ept_1.data'), 'rb') as f:
            ds = pickle.load(f)
        ds = ds[0:41]
        plt.figure()
        ax = plt.figure().gca()
        for d in ds:
            h = [e[0] for e in d]
            p = [e[1] for e in d]
            plt.plot(h, p, alpha=0.9)

        ax.grid(b=True, axis='y')
        ax.grid(b=False, axis='x')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel("Feasible ratio (%)")
        plt.xlabel("H")
        # plt.title("Feasible ratio comparison on {}".format(cname))
        plt.legend(INIT_METHODS_LEGEND, frameon=True)
        out_filepath = join(out_test_dir, 'feasible_ratio.png')
        plt.savefig(out_filepath, dpi=400)
        plt.close('all')

    out_dir = 'results/ept_init/ept_1_2'

    test_path = './data/ept_init'
    for testname in os.listdir(test_path):
        if 'dem' not in testname or (testnames is not None and all(e not in testname for e in testnames)):
            continue
        basename = os.path.splitext(testname)[0]
        out_test_dir = join(out_dir, basename)
        os.makedirs(out_test_dir, exist_ok=True)
        filepath = join(test_path, testname)

        if not os.path.isfile(os.path.join(out_test_dir, 'done_ept_1.flag')) or RERUN:
            run(filepath, out_test_dir)

        plot(out_test_dir, testname)


def run_ept_2(testnames=None):
    out_dir = 'results/ept_init/ept_1_2'
    test_path = './data/ept_init'

    def run(filepath, out_test_dir, max_hop1, max_hop2):
        print(f"Running: {filepath}")
        data_1 = []
        data_2 = []
        label_1 = []
        label_2 = []
        for i in range(len(INIT_METHODS)):
            n_seed = 1 if TESTING else 10
            pop_size = 20 if TESTING else 100
            x1, x2 = [], []
            nos1 = 0
            nos2 = 0
            for seed in range(1, n_seed+1):
                sol1 = init_population(
                    filepath, INIT_METHODS[i], pop_size, max_hop1, seed)
                nos1 += sum( a != float('inf') and b != float('inf') for a, b in sol1 )
                x1.extend([e[0] for e in sol1 if e[0] != float('inf')])

                sol2 = init_population(
                    filepath, INIT_METHODS[i], pop_size, max_hop2, seed)
                nos2 += sum( a != float('inf') and b != float('inf') for a, b in sol2 )
                x2.extend([e[0] for e in sol2 if e[0] != float('inf')])
            avs1, avs2 = nos1 / n_seed, nos2 / n_seed
            label_1.append(int(100*avs1/pop_size))
            label_2.append(int(100*avs2/pop_size))
            data_1.append(x1)
            data_2.append(x2)

        with open(join(out_test_dir, 'ept_2.data'), 'wb') as f:
            pickle.dump((data_1, data_2, label_1, label_2), f)
        open(os.path.join(out_test_dir, 'done_ept_2.flag'), 'a').close()

    def plot(out_test_dir, max_hop1, max_hop2, outname):

        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)
            plt.setp(bp['fliers'], color='red', marker='+')

        data_1, data_2, label_1, label_2 = None, None, None, None
        with open(join(out_test_dir, 'ept_2.data'), 'rb') as f:
            data_1, data_2, label_1, label_2 = pickle.load(f)

        plt.style.use('seaborn-white')
        plt.grid(True)
        fig, ax = plt.subplots()
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                       alpha=0.5)

        box_colors = ['blue', 'red']

        pos = [i for i in range(1, len(data_1)*3+1, 3)]
        bp = plt.boxplot(data_1, positions=pos, widths=0.6, notch=False)
        set_box_color(bp, box_colors[0])

        
        pos = [i for i in range(2, len(data_1)*3+2, 3)]
        bp = plt.boxplot(data_2, positions=pos, widths=0.6, notch=False)
        set_box_color(bp, box_colors[1])

        ax.set_xticklabels(INIT_METHODS_LEGEND)
        pos = [i+0.5 for i in range(1, len(INIT_METHODS)*3+1, 3)]
        ax.set_xticks(pos)
        merged = list(itertools.chain.from_iterable(data_1 + data_2))
        max_relays = max(merged)
        plt.ylim(top=max_relays + 1)

        ax.set(
            axisbelow=True,  # Hide the grid behind plot objects
            title='The comparison of relays distribution on different initializations',
            xlabel='Initializations',
            ylabel='Number of used relays',
        )

        upper_labels = (label_1, label_2)
        weights = ['bold', 'semibold']


        pos = [i for i in range(1, len(data_1)*3, 1) if i % 3 != 0]
        for tick in range(len(data_1)*2):
            k = tick % 2
            plt.text(pos[tick], .95, '{}%'.format(upper_labels[tick%2][tick//2]),
                     transform=ax.get_xaxis_transform(),
                     horizontalalignment='center', size='x-small',
                     weight=weights[k], color=box_colors[k])

        # plt.title(outname)

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c='blue', label=f'H = {max_hop1}')
        plt.plot([], c='red', label=f'H = {max_hop2}')
        ax.legend(loc=3, frameon=True)

        ax.grid(b=True, axis='y')
        ax.grid(b=False, axis='x')

        out_filepath = join(out_test_dir, outname)
        plt.savefig(out_filepath, dpi=400)
        plt.close('all')

    for testname in os.listdir(test_path):
        if 'dem' not in testname or (testnames is not None and all(e not in testname for e in testnames)):
            continue
        basename = os.path.splitext(testname)[0]
        out_test_dir = join(out_dir, basename)
        os.makedirs(out_test_dir, exist_ok=True)
        filepath = join(test_path, testname)
        max_hop1, max_hop2 = (6, 10) if TESTING else (16, 30)
        outname = 'relays\' distribution'
        if not os.path.isfile(os.path.join(out_test_dir, 'done_ept_2.flag')) or RERUN:
            run(filepath, out_test_dir, max_hop1, max_hop2)
        plot(out_test_dir, max_hop1, max_hop2, outname)


def run_ept_3(testnames=None):
    def run_solver(solver, model, max_hop, out_dir):
        config = load_config(CONFIG_FILE, model)

        model_dict = {}
        test_path = './data/ept_init'
        for i in range(len(INIT_METHODS)):
            smodel = '{}.{}'.format(model, i)
            out_model_dir = os.path.join(out_dir, smodel)
            os.makedirs(out_model_dir, exist_ok=True)
            config['encoding']['init_method'] = INIT_METHODS[i]
            config['models']['gens'] = 2 if TESTING else 200
            config['data']['max_hop'] = max_hop
            config['algorithm']['pop_size'] = 20 if TESTING else 100
            print(config)
            model_dict[smodel] = smodel
            run.run_solver(solver, smodel, test_path, output_dir=out_model_dir, overwrite=RERUN,
                           testnames=testnames, save_history=True, config=config, seed=42)

        marker = ['>', (5,0), (5,1), (5,2), '+', 'o'] 
        marker.reverse()
        summarization.summarize_model(
            model_dict, working_dir=out_dir, 
            cname=f'sum-{model}', 
            s=20, 
            marker=marker, 
            plot_line=True, 
            linewidth=0.8,
            linestyle='dashed')
        return model_dict

    def plot(all_model_dict, out_dir):
        pass

    out_dir = 'results/ept_init/ept_3'
    g = 0 if TESTING else 1
    max_hops = [6, 10] if TESTING else [10, 20]
    all_model_dict = {}
    for h, max_hop in enumerate(max_hops):
        all_model_dict[max_hop] = {}
        md = run_solver(solver_mhn_gprim3, f'{g}.2.8.0.{h}', max_hop, out_dir)
        all_model_dict[max_hop]['guided prim'] = md 
        md = run_solver(solver_mhn_gprim2, f'{g}.2.7.0.{h}', max_hop, out_dir)
        all_model_dict[max_hop]['guided prim'] = md 
        md = run_solver(solver_mhn_kruskal, f'{g}.2.2.0.{h}', max_hop, out_dir)
        all_model_dict[max_hop]['kruskal'] = md 
        md = run_solver(solver_mhn_nrk, f'{g}.2.1.0.{h}', max_hop, out_dir)
        all_model_dict[max_hop]['netkeys'] = md 
        md = run_solver(solver_mhn_prim, f'{g}.2.4.0.{h}', max_hop, out_dir)
        all_model_dict[max_hop]['prim'] = md 
        md = run_solver(solver_mhn_prufer, f'{g}.2.6.0.{h}', max_hop, out_dir)
        all_model_dict[max_hop]['prufer'] = md 


    plot(all_model_dict, out_dir)

def run_ept_4(testnames=None):
    out_dir = 'results/ept_init/ept_3'
    max_hops = [6, 10] if TESTING else [10, 20]
    g = 0 if TESTING else 1

    for h, max_hops in enumerate(max_hops):
        for i, init_method in enumerate(INIT_METHODS):
            model_dict = {
                'guided prim3': f'{g}.2.8.0.{h}.{i}',
                'guided prim2': f'{g}.2.7.0.{h}.{i}',
                'kruskal': f'{g}.2.2.0.{h}.{i}',
                'netkeys': f'{g}.2.1.0.{h}.{i}',
                'prim': f'{g}.2.4.0.{h}.{i}',
                'prufer': f'{g}.2.6.0.{h}.{i}',
            }
            summarization.summarize_model( model_dict, working_dir=out_dir, cname=f'sum-{g}.{h}.{i}')

if __name__ == '__main__':
    testname = 'test' if TESTING else 'medium'
    testnames = [testname]
    run_ept_1(testnames)
    # run_ept_2(testnames)
    # run_ept_3(testnames)
    # run_ept_4(testnames)
