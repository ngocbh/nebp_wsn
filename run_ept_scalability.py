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
import solver_mhn_prufer
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
TESTING=True


def run_ept():
    def run_solver(solver, model_name, model):
        config = load_config(CONFIG_FILE, model)
        out_dir = 'results/ept_scalability'

        model_dict = {}
        test_path = './data/ept_scalability'

        out_model_dir = os.path.join(out_dir, model)
        os.makedirs(out_model_dir, exist_ok=True)
        print(config)
        model_dict[model_name] = model
        run.run_solver(solver, model, test_path, output_dir=out_model_dir, save_history=False, config=config, seed=42)

        summarization.summarize_model(model_dict, working_dir=out_dir, cname=f'sum-{model}')

    g = 0 if TESTING else 1
    run_solver(solver_mhn_gprim, 'guided prim', f'{g}.2.5.0')
    run_solver(solver_mhn_kruskal, 'kruskal', f'{g}.2.2.0')
    run_solver(solver_mhn_nrk, 'netkeys', f'{g}.2.1.0')
    run_solver(solver_mhn_prim, 'prim', f'{g}.2.4.0')
    run_solver(solver_mhn_prufer, 'prufer', f'{g}.2.6.0')

if __name__ == '__main__':
    run_ept()
