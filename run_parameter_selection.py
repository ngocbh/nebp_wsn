from __future__ import absolute_import

from utils.configurations import load_config, gen_output_dir

import solver_mhn_kruskal
import solver_mhn_gprim4
import solver_mhn_nrk
import solver_mhn_prim
import solver_mhn_prufer
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

def choose_pc(solver, model, tests, pc_list, default_pm=0.1, gprim=False):
    pm = default_pm
    config = load_config(CONFIG_FILE, model)
    out_dir = 'results/params_selection'

    model_dict = {}
    test_path = './data/params_selection'
    for i in range(len(pc_list)):
        smodel = '{}.{}.{}'.format(model, i, 1) 
        out_model_dir = os.path.join(out_dir, smodel)
        if not gprim:
            config['encoding']['cro_prob'] = pc_list[i]
            config['encoding']['mut_prob'] = pm
        else:
            config['encoding']['cro_prob'] = pc_list[i]
            config['encoding']['mut_prob_a'] = pm[0]
            config['encoding']['mut_prob_b'] = pm[1]
        model_dict[smodel] = smodel
        run.run_solver(solver, smodel, test_path, out_model_dir, tests, save_history=False, config=config)

    summarization.summarize_model(model_dict, working_dir=out_dir, cname=f'sum-pc-{model}')
        
def choose_pm(solver, model, tests, pm_list, default_pc=0.7, gprim=False):
    pc = default_pc
    config = load_config(CONFIG_FILE, model)
    out_dir = 'results/params_selection'

    model_dict = {}
    test_path = './data/params_selection'
    for i in range(len(pm_list)):
        smodel = '{}.{}.{}'.format(model, 4, i) 
        out_model_dir = os.path.join(out_dir, smodel)
        if not gprim:
            config['encoding']['cro_prob'] = pc
            config['encoding']['mut_prob'] = pm_list[i]
        else:
            config['encoding']['cro_prob'] = pc
            config['encoding']['mut_prob_a'] = pm_list[i][0]
            config['encoding']['mut_prob_b'] = pm_list[i][1]
        model_dict[smodel] = smodel
        run.run_solver(solver, smodel, test_path, out_model_dir, tests, save_history=False, config=config)

    summarization.summarize_model(model_dict, working_dir=out_dir, cname=f'sum-pm-{model}')

def choose_parameters(solver, model, tests, gprim=False):
    if not gprim:
        pc_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        choose_pc(solver, model, tests, pc_list)
        pm_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
        choose_pm(solver, model, tests, pm_list)
    else:
        pc_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        choose_pc(solver, model, tests, pc_list, default_pm=(0.9,0.1), gprim=gprim)
        pm_list = [(0.9, 0.1), (0.8, 0.2), (0.8, 0.1), (0.7, 0.2)]
        choose_pm(solver, model, tests, pm_list, default_pc=0.5, gprim=gprim)

if __name__ == '__main__':
    # tests = random_tests() 
    tests = ['']
    choose_parameters(solver_mhn_gprim4, "1.6.9.0", tests)
    choose_parameters(solver_mhn_kruskal, "1.6.2.0", tests)
    choose_parameters(solver_mhn_prim, "1.6.4.0", tests)
    choose_parameters(solver_mhn_prufer, "1.6.6.0", tests)
    choose_parameters(solver_mhn_nrk, "1.6.1.0", tests)
    summarization.average_tests_score('./results/params_selection/sum-pc-1.6.1.0')
    summarization.average_tests_score('./results/params_selection/sum-pc-1.6.2.0')
    summarization.average_tests_score('./results/params_selection/sum-pc-1.6.4.0')
    summarization.average_tests_score('./results/params_selection/sum-pc-1.6.9.0')
    summarization.average_tests_score('./results/params_selection/sum-pc-1.6.6.0')
    summarization.average_tests_score('./results/params_selection/sum-pm-1.6.1.0')
    summarization.average_tests_score('./results/params_selection/sum-pm-1.6.2.0')
    summarization.average_tests_score('./results/params_selection/sum-pm-1.6.4.0')
    summarization.average_tests_score('./results/params_selection/sum-pm-1.6.9.0')
    summarization.average_tests_score('./results/params_selection/sum-pm-1.6.6.0')
