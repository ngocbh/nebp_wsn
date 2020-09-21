from __future__ import absolute_import

from utils.configurations import load_config, gen_output_dir

import solver_mhn_kruskal
import solver_mhn_gprim
import solver_mhn_nrk
import solver_mhn_prim
import summarization
import run

import os
from os.path import join
from random import Random
import pandas as pd

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(WORKING_DIR, './configs/_configurations.yml')
DATA_DIR = os.path.join(WORKING_DIR, "./data/small/multi_hop")

def random_tests():
    tests = []
    rand = Random(42)
    for filename in os.listdir(DATA_DIR):
        if 'dem' in filename:
            tests.append(filename)

    return rand.sample(tests, 10)

def sum_score(working_dir):
    metric_sum = None
    n_tests = 0
    for test in os.listdir(working_dir):
        if 'dem' in test:
            test_dir = os.path.join(working_dir, test)
            test_df = pd.read_csv(join(test_dir, 'metrics_comparison.csv'))
            models = test_df['models']
            test_df = test_df.drop(columns='models')
            if metric_sum is None:
                metric_sum = test_df
            elif metric_sum.shape != test_df.shape:
                raise ValueError(f'difference metric shape on test {test}')
            else:
                metric_sum = metric_sum.add(test_df, fill_value=0)
            n_tests += 1
    metric_sum = metric_sum.div(n_tests)
    metric_sum.insert(0, 'models', models, True)
    out_file = join(working_dir, 'sum_test_scores.csv')
    metric_sum.to_csv(out_file, index=False)


def choose_pc(solver, model, tests):
    pc_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pm = 0.1
    config = load_config(CONFIG_FILE, model)
    out_dir = 'results/small/multi_hop/parsec'

    model_dict = {}
    test_path = './data/small/multi_hop'
    for i in range(len(pc_list)):
        smodel = '{}.{}.{}'.format(model, i, 1) 
        out_model_dir = os.path.join(out_dir, smodel)
        config['encoding']['cro_prob'] = pc_list[i]
        config['encoding']['mut_prob'] = pm
        model_dict[smodel] = smodel
        run.run_solver(solver, smodel, test_path, out_model_dir, tests, save_history=False, config=config)

    summarization.summarize_model(model_dict, working_dir=out_dir, cname=f'sum-pc-{model}')
        
def choose_pm(solver, model, tests):
    pc = 0.9
    pm_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    config = load_config(CONFIG_FILE, model)
    out_dir = 'results/small/multi_hop/parsec'

    model_dict = {}
    test_path = './data/small/multi_hop'
    for i in range(len(pm_list)):
        smodel = '{}.{}.{}'.format(model, 4, i) 
        out_model_dir = os.path.join(out_dir, smodel)
        config['encoding']['cro_prob'] = pc
        config['encoding']['mut_prob'] = pm_list[i]
        model_dict[smodel] = smodel
        run.run_solver(solver, smodel, test_path, out_model_dir, tests, save_history=False, config=config)

    summarization.summarize_model(model_dict, working_dir=out_dir, cname=f'sum-pm-{model}')

def choose_parameters(solver, model, tests):
    choose_pc(solver, model, tests)
    choose_pm(solver, model, tests)

if __name__ == '__main__':
    # tests = random_tests() 
    # choose_parameters(solver_mhn_gprim, "1.0.5.0", tests)
    # choose_parameters(solver_mhn_kruskal, "1.0.2.0", tests)
    # choose_parameters(solver_mhn_nrk, "1.0.1.0", tests)
    # choose_parameters(solver_mhn_prim, "1.0.4.0", tests)
    sum_score('./results/small/multi_hop/parsec/sum-pc-1.0.1.0')
    sum_score('./results/small/multi_hop/parsec/sum-pc-1.0.2.0')
    sum_score('./results/small/multi_hop/parsec/sum-pc-1.0.4.0')
    sum_score('./results/small/multi_hop/parsec/sum-pc-1.0.5.0')
    sum_score('./results/small/multi_hop/parsec/sum-pm-1.0.1.0')
    sum_score('./results/small/multi_hop/parsec/sum-pm-1.0.2.0')
    sum_score('./results/small/multi_hop/parsec/sum-pm-1.0.4.0')
    sum_score('./results/small/multi_hop/parsec/sum-pm-1.0.5.0')
