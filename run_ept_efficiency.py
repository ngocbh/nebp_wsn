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

RERUN=False
TESTING=True


def run_ept():
    def plot():
        pass

    ept = 0 if TESTING else 1
    input_dir = './data/test' if TESTING else './data/ept_efficiency'
    output_dir = None
    testset = 0 if TESTING else 3
    testnames = 'test' if TESTING else ''
    k = 5
    config = None
    if TESTING:
        config = {'models': {}, 'algorithm': {}}
        config['models']['gens'] = 2
        config['algorithm']['pop_size'] = 10
        config['algorithm']['selection_size'] = 10

    sum_list = run.run_mhn_experiment(ept, input_dir, output_dir, testset, testnames, k, overwrite=RERUN, config=config)

if __name__ == '__main__':
    run_ept()
