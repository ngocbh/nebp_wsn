"""
File: run.py
Author: ngocjr7
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

from __future__ import absolute_import

from parameters import SHP
from parameters import MHP

import single_hop_nsgaii
import multi_hop_nsgaii

import os
import joblib

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
OVERWRITE = False

def is_done(filename, output_dir):
    filebase, _ = os.path.splitext(os.path.basename(filename))
    dirname = os.path.join(output_dir, filebase)
    if not os.path.isdir(dirname):
        return False
    if os.path.isfile(os.path.join(dirname, 'done.flag')):
        return True
    return False


def run_single_hop_problem(model, input_dir, output_dir):
    print(f"Running single hop problem on model {model}")
    shp = SHP()
    shp.load_model(model)
    print(shp.GENS)
    datapath = os.path.join(WORKING_DIR, input_dir)

    test_list = []
    done_list = []

    for file in os.listdir(datapath):
        if 'dem' not in file:
            continue
        filepath = os.path.join(datapath, file)
        if not is_done(file, output_dir) or OVERWRITE:
            test_list.append(filepath)
        else:
            done_list.append(filepath)

    print("Done those tests: ")
    print(done_list)
    print("Run those tests: ")
    print(test_list)

    joblib.Parallel(n_jobs=-1)(joblib.delayed(single_hop_nsgaii.solve)
                               (file, output_dir=output_dir, visualization=True, shp=shp) for file in test_list)


def run_multi_hop_problem(model, input_dir, output_dir):
    print(f"Running multi-hop problem on model {model}")
    mhp=MHP()
    mhp.load_model(model)
    datapath = os.path.join(WORKING_DIR, input_dir)

    test_list = []
    done_list = []

    for file in os.listdir(datapath):
        if 'dem' not in file:
            continue
        filepath = os.path.join(datapath, file)
        if not is_done(file, output_dir) or OVERWRITE:
            test_list.append(filepath)
        else:
            done_list.append(filepath)

    print(f"Done {len(done_list)} tests: ")
    print(done_list)
    print(f"Working on {len(test_list)} tests: ")
    print(test_list)

    joblib.Parallel(n_jobs=-1)(joblib.delayed(multi_hop_nsgaii.solve)(
        file, output_dir=output_dir, visualization=True, mhp=mhp) for file in test_list)


if __name__ == "__main__":
    # print("Running Test Model...")
    # run_single_hop_problem("test", 'data/medium/single_hop', 'results/medium/test/single_hop')
    # run_multi_hop_problem("test", 'data/medium/multi_hop', 'results/medium/test/multi_hop')
    print("Running 0.0.1 model")
    print("Running small data")
    run_single_hop_problem('0.0.1','data/small/single_hop', 'results/small/0.0.1/single_hop')
    print("Running medium data")
    run_single_hop_problem('0.0.1', 'data/medium/single_hop', 'results/medium/0.0.1/single_hop')
    run_multi_hop_problem('0.0.1', 'data/medium/multi_hop', 'results/medium/0.0.1/multi_hop')
    print("Running 0.0.2 model")
    run_single_hop_problem('0.0.2', 'data/medium/single_hop', 'results/medium/0.0.2/single_hop')
    run_multi_hop_problem('0.0.2', 'data/medium/multi_hop', 'results/medium/0.0.2/multi_hop')
    
