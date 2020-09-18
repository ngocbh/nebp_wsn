from __future__ import absolute_import

from utils.configurations import load_config, gen_output_dir

import solver_mhn_kruskal
import solver_mhn_gprim
import solver_mhn_nrk
import solver_mhn_prim
import summarization

import os
from random import Random

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

def choose_pc(solver, model):
    pc_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pm = 0.1
    config = load_config(CONFIG_FILE, model)
    out_dir = 'results/small/multi_hop/parsec'

    tests = random_tests() 
    model_dict = {}
    for testname in tests:
        test_path = os.path.join('./data/small/multi_hop', testname)
        for i in range(len(pc_list)):
            out_test_dir = os.path.join(out_dir, smodel)
            smodel = '{}.{}.{}'.format(model, i, 1) 
            config['encoding']['cro_prob'] = pc_list[i]
            config['encoding']['mut_prob'] = pm
            model_dict[smodel] = smodel
            solver.solve(test_path, out_dir=out_test_dir, model=smodel,config=config, save_history=False)

    summarization.summarize_model(model_dict, working_dir=out_dir, cname='parsec')
        


if __name__ == '__main__':
    choose_pc(solver_mhn_nrk, "1.0.2.0")
