"""
File: solver_mhn_edgesets_naive.py
Created by ngocjr7 on 2020-08-23 00:55
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

from geneticpython.tools.visualization import save_history_as_gif, visualize_fronts
from geneticpython.engines import NSGAIIEngine
from geneticpython.models.tree import NetworkRandomKeys
from geneticpython import Population
from geneticpython.core.operators import TournamentSelection, SBXCrossover, PolynomialMutation

from utils.configurations import load_config, gen_output_dir
from utils import WusnInput
from utils import visualize_front, make_gif, visualize_solutions, remove_file, save_results
from problems import MultiHopProblem
from networks import MultiHopNetwork
from edge_sets import WusnEdgeSets

import os
import sys
import matplotlib
import time

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(WORKING_DIR, './configs/_configurations.yml')

def check_config(config, filename, model):
    if config['data']['name'] not in filename:
        raise ValueError('Model {} is used for {}, file {} is not'.format(model, config['data'], filename))
    if config['encoding']['name'] != 'netkeys':
        raise ValueError('encoding {} != {}'.format(config['encoding']['name'], 'netkeys'))
    if config['algorithm']['name'] != 'nsgaii':
        raise ValueError('algorithm {} != {}'.format(config['algorithm']['name'], 'nsgaii'))

def solve(filename, output_dir=None, model='0.0.0.0'):
    start_time = time.time()

    config = load_config(CONFIG_FILE, model)
    check_config(config, filename, model)
    output_dir = output_dir or gen_output_dir(filename, model)

    basename, _ = os.path.splitext(os.path.basename(filename))
    os.makedirs(os.path.join(
        WORKING_DIR, '{}/{}'.format(output_dir, basename)), exist_ok=True)
    print(basename)

    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    problem = MultiHopProblem(inp, config['data']['max_hop'])
    network_temp = MultiHopNetwork(problem)
    indv_temp = WusnEdgeSets(problem, network_temp)



if __name__ == '__main__':
    solve('data/small/multi_hop/ga-dem1_r25_1_0.json', model = '0.0.0.0')
