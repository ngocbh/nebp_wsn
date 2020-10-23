"""
File: run.py
Author: ngocjr7
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

from __future__ import absolute_import

from collections import OrderedDict

import solver_mhn_nrk
import solver_mhn_gprim
import solver_mhn_gprim2
import solver_mhn_gprim3
import solver_mhn_gprim4
import solver_mhn_kruskal
import solver_mhn_prim
import solver_mhn_prufer
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


def run_solver(solver, model, input_dir, output_dir=None, testnames=None, overwrite=False, **kwargs):
    print(f"Running multi-hop problem on model {model}")
    datapath = os.path.join(WORKING_DIR, input_dir)

    test_list = []
    done_list = []

    print(testnames)
    output_dir = output_dir or gen_output_dir(input_dir, model)
    for file in os.listdir(datapath):
        if ('dem' not in file) or (testnames is not None and all(e not in file for e in testnames)):
            continue
        filepath = os.path.join(datapath, file)
        if not is_done(file, output_dir) or overwrite:
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
        run_solver(solver, smodel, input_dir=input_dir,
                   output_dir=output_dir, testnames=testnames, seed=seed, **kwargs)
    return model_list


def run_mhn_experiment(ept, 
                       input_dir, 
                       output_dir=None, 
                       testset=0, 
                       testnames=None, 
                       k=10, 
                       overwrite=False, 
                       config=None, 
                       referenced=False,
                       referenced_dir=None,
                       summ=True,
                       **kwargs):
    print("Running guided prim solver...")
    output_dir = output_dir or input_dir.replace('data', 'results')
    gprim_model = f'{ept}.{testset}.8.0'
    gprim_model_list = multi_run_solver(solver_mhn_gprim3,
                                        model=gprim_model,
                                        input_dir=input_dir,
                                        k=k,
                                        testnames=testnames,
                                        save_history=False,
                                        overwrite=overwrite,
                                        config=config)

    print("Running guided prim 4 solver...")
    output_dir = input_dir.replace('data', 'results')
    gprim4_model = f'{ept}.{testset}.9.0'
    gprim4_model_list = multi_run_solver(solver_mhn_gprim4,
                                        model=gprim4_model,
                                        input_dir=input_dir,
                                        k=k,
                                        testnames=testnames,
                                        save_history=False,
                                        overwrite=overwrite,
                                        config=config)

    print("Running kruskal solver...")
    kruskal_model = f'{ept}.{testset}.2.0'
    kruskal_model_list = multi_run_solver(solver_mhn_kruskal,
                                          model=kruskal_model,
                                          input_dir=input_dir,
                                          k=k,
                                          testnames=testnames,
                                          save_history=False,
                                          overwrite=overwrite,
                                          config=config)

    print("Running prim solver...")
    prim_model = f'{ept}.{testset}.4.0'
    prim_model_list = multi_run_solver(solver_mhn_prim,
                                       model=prim_model,
                                       input_dir=input_dir,
                                       k=k,
                                       testnames=testnames,
                                       save_history=False,
                                       overwrite=overwrite,
                                       config=config)

    print("Running netkeys solver...")
    netkeys_model = f'{ept}.{testset}.1.0'
    netkeys_model_list = multi_run_solver(solver_mhn_nrk,
                                          model=netkeys_model,
                                          input_dir=input_dir,
                                          k=k,
                                          testnames=testnames,
                                          save_history=False,
                                          overwrite=overwrite,
                                          config=config)

    print("Running prufer solver...")
    prufer_model = f'{ept}.{testset}.6.0'
    prufer_model_list = multi_run_solver(solver_mhn_prufer,
                                         model=prufer_model,
                                         input_dir=input_dir,
                                         k=k,
                                         testnames=testnames,
                                         save_history=False,
                                         overwrite=overwrite,
                                         config=config)

    if not summ:
        return 

    print("Summarizing...")
    summarization_list = []
    for i in range(k):
        model_dict = OrderedDict()
        model_dict['A'] = prufer_model_list[i]
        model_dict['B'] = netkeys_model_list[i]
        model_dict['C'] = prim_model_list[i]
        model_dict['D'] = kruskal_model_list[i]
        model_dict['E'] = gprim_model_list[i]
        model_dict['F'] = gprim4_model_list[i]
        cname = f'summarization_{i+1}'
        summarization_list.append(cname)
        summarization.summarize_model(
            model_dict, output_dir, cname, testnames,
            referenced=referenced, referenced_dir=referenced_dir, **kwargs)

    summarization.calc_average_metrics(
        summarization_list, output_dir, f'avarage1-{k}', testnames, referenced=referenced)

    return summarization_list


if __name__ == "__main__":
    # print("Running Test Model...")
    testnames = {'ga-dem1_r25_1_0.json', 'ga-dem1_r25_1_40.json'}
    run_mhn_experiment(1, './data/small/multi_hop',
                       testset=0, testnames=None, k=10)
