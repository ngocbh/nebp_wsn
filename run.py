"""
File: run.py
Author: ngocjr7
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

from __future__ import absolute_import

import solver_mhn_nrk
import solver_mhn_gprim
import solver_mhn_kruskal
import solver_mhn_prim
import summarization

import os
import joblib

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
OVERWRITE = False

def gen_output_dir(input_dir, model):
    output_dir = input_dir.replace('data', 'results')
    output_dir = os.path.join(output_dir, model)
    return output_dir

def is_done(filename, output_dir):
    filebase, _ = os.path.splitext(os.path.basename(filename))
    dirname = os.path.join(output_dir, filebase)
    if not os.path.isdir(dirname):
        return False
    if os.path.isfile(os.path.join(dirname, 'done.flag')):
        return True
    return False

def run_solver(solver, model, input_dir, output_dir=None, testnames=None, **kwargs):
    print(f"Running multi-hop problem on model {model}")
    datapath = os.path.join(WORKING_DIR, input_dir)

    test_list = []
    done_list = []

    print(testnames)
    output_dir = output_dir or gen_output_dir(input_dir, model)
    for file in os.listdir(datapath):
        if 'dem' not in file or (testnames is not None and file not in testnames):
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

    joblib.Parallel(n_jobs=-1)(joblib.delayed(solver.solve)(
        file, output_dir=output_dir, model=model, **kwargs) for file in test_list)

def multi_run_solver(solver, model, input_dir, k, output_dir=None, testnames=None, **kwargs):
    model_list = []
    for seed in range(1, k+1):
        smodel = '{}.{}'.format(model, seed)
        model_list.append(smodel)
        run_solver(solver, smodel, input_dir=input_dir, output_dir=output_dir, testnames=testnames, seed=seed, **kwargs)
    return model_list

def run_mhn_experiment(ept, input_dir, testset=0, testnames=None, k=10):
    output_dir = input_dir.replace('data', 'results') 
    gprim_model = f'{ept}.{testset}.5.0'
    gprim_model_list = multi_run_solver(solver_mhn_gprim, 
                                        model=gprim_model, 
                                        input_dir=input_dir, 
                                        k=k, 
                                        testnames=testnames,
                                        save_history=False)

    kruskal_model = f'{ept}.{testset}.2.0' 
    kruskal_model_list = multi_run_solver(solver_mhn_kruskal, 
                                         model=kruskal_model, 
                                         input_dir=input_dir, 
                                         k=k,
                                         testnames=testnames,
                                         save_history=False)

    prim_model = f'{ept}.{testset}.4.0'
    prim_model_list = multi_run_solver(solver_mhn_prim, 
                                         model=prim_model, 
                                         input_dir=input_dir, 
                                         k=k,
                                         testnames=testnames,
                                         save_history=False)

    netkeys_model = f'{ept}.{testset}.1.0'
    netkeys_model_list = multi_run_solver(solver_mhn_nrk, 
                                         model=netkeys_model, 
                                         input_dir=input_dir, 
                                         k=k,
                                         testnames=testnames,
                                         save_history=False)

    summarization_list = []
    for i in range(k):
        model_dict = {}
        model_dict['netkeys'] = netkeys_model_list[i]
        model_dict['prim'] = prim_model_list[i]
        model_dict['kruskal'] = kruskal_model_list[i]
        model_dict['guided prim'] = gprim_model_list[i]
        cname = f'summarization_{i+1}'
        summarization_list.append(cname)
        summarization.summarize_model(model_dict, output_dir, cname, testnames)


    summarization.calc_average_metrics(summarization_list, output_dir, f'avarage1-{k}', testnames)


if __name__ == "__main__":
    # print("Running Test Model...")
    testnames = {'ga-dem1_r25_1_0.json', 'ga-dem1_r25_1_40.json'}
    run_mhn_experiment(0, './data/small/multi_hop', testset=0, testnames=testnames, k=2)
    
