"""
File: summarization.py
Created by ngocjr7 on 2020-09-08 10:51
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""
from geneticpython.tools.visualization import visualize_fronts
from geneticpython.tools.analyzers.pareto_metrics import delta_metric, coverage_metric, spacing_metric, non_dominated_solutions
from yaml import Loader
from os.path import join

import json
import os
import yaml
import pandas as pd

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


def read_pareto(filepath):
    data = json.load(open(filepath, mode='r'))
    pareto = set()
    for solution in data:
        objectives = (solution['num_used_relays'], solution['energy_consumption'])
        pareto.add(objectives)
    pareto = list(pareto)
    return pareto

def visualize_test(pareto_dict, output_dir, show=True):
    filepath = os.path.join(output_dir, 'front_comparison.png')
    visualize_fronts(pareto_dict, 
                     filepath=filepath,
                     title='pareto fronts comparison', 
                     objective_name=['used relays', 'energy'], 
                     save=True, show=show)

def summarize_metrics(pareto_dict, output_dir):
    metrics = {}
    metrics['models'] = list(pareto_dict.keys())
    metrics['delta'] = []
    metrics['spacing'] = []
    metrics['nds'] = []
    for key in pareto_dict.keys():
        metrics['c_' + key] = []
    metrics['score'] = []

    n = len(pareto_dict.keys())
    c_matrix = [[] for _ in range(n)]
    i = 0

    for name, pareto in pareto_dict.items():
        if name != metrics['models'][i]:
            raise ValueError("Summarize metrics error")

        metrics['delta'].append(delta_metric(pareto))
        metrics['spacing'].append(spacing_metric(pareto))
        metrics['nds'].append(non_dominated_solutions(pareto))
        for other_name, other_pareto in pareto_dict.items():
            c = coverage_metric(pareto, other_pareto)
            metrics['c_' + other_name].append(c)
            c_matrix[i].append(c)

        i += 1
    
    for i in range(n):
        score = 0
        for j in range(n):
            if c_matrix[i][j] - c_matrix[j][i] > 0:
                score += 1
        metrics['score'].append(score)

    df = pd.DataFrame(data=metrics)
    filepath = os.path.join(output_dir, 'metrics_comparison.csv')
    df.to_csv(filepath, index=False)

def summarize_test(testname, model_dict, working_dir, cname):
    absworking_dir = os.path.join(WORKING_DIR, working_dir)
    pareto_dict = {}
    config_dict = {}
    for name, model in model_dict.items():
        model_dir = os.path.join(absworking_dir, model)
        test_dir = os.path.join(model_dir, testname)
        if not os.path.isdir(test_dir):
            continue
        pareto = read_pareto(os.path.join(test_dir, 'pareto-front.json'))
        pareto_dict[name] = pareto
        config = yaml.load(open(os.path.join(test_dir, '_config.yml')), Loader=Loader)
        config["model_name"] = model
        config_dict[name] = config
    
    output_dir = os.path.join(absworking_dir, cname)
    out_test_dir = os.path.join(output_dir, testname)
    os.makedirs(out_test_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config_comparison.yml'), mode='w') as f:
        f.write(yaml.dump(config_dict))

    visualize_test(pareto_dict, output_dir=out_test_dir, show=False)
    summarize_metrics(pareto_dict, output_dir=out_test_dir)


def summarize_model(model_dict, working_dir, cname=None, testnames=None):
    tests = set()
    absworking_dir = os.path.join(WORKING_DIR, working_dir)
    
    comparison_name = "sum"
    for model in model_dict.values():
        comparison_name += '+' + model
        model_dir = os.path.join(absworking_dir, model)
        for filename in os.listdir(model_dir):
            inp_name = f'{filename}.json'
            if ('dem' in filename or 'test' in filename) and \
                    (testnames is None or inp_name in testnames):
                tests.add(filename)

    cname = cname or comparison_name
    output_dir = os.path.join(absworking_dir, cname)
    os.makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, 'model_dict.json'), mode='w') as f:
        f.write(json.dumps(model_dict, indent=4))

    for test in tests:
        summarize_test(test, model_dict, working_dir, cname)


def calc_average_metrics(summarization_list, working_dir, cname, testnames=None):
    all_tests = set()
    absworking_dir = os.path.join(WORKING_DIR, working_dir)
    for model in summarization_list:
        model_dir = os.path.join(absworking_dir, model)
        for filename in os.listdir(model_dir):
            inp_name = f'{filename}.json'
            if ('dem' in filename or 'test' in filename) and \
                    (testnames is None or inp_name in testnames):
                all_tests.add(filename)

    output_dir = os.path.join(absworking_dir, cname)
    os.makedirs(output_dir, exist_ok=True)

    feasible_tests = all_tests.copy()
    for model in summarization_list:
        model_dir = os.path.join(absworking_dir, model)
        model_tests = set()
        for filename in os.listdir(model_dir):
            inp_name = f'{filename}.json'
            if ('dem' in filename or 'test' in filename) and \
                    (testnames is None or inp_name in testnames):
                model_tests.add(filename)
        feasible_tests.intersection(model_tests)

    with open(os.path.join(output_dir, 'status.txt'), mode='w') as f:
        f.write('Feasible_tests: ')
        f.write(str(feasible_tests))
        f.write('\nUnsolved_tests: ' + str(all_tests.difference(feasible_tests)))

    for test in feasible_tests:
        metric_sum = None
        models = None
        for summ in summarization_list:
            test_dir = join(join(absworking_dir, summ), test)
            test_df = pd.read_csv(join(test_dir, 'metrics_comparison.csv'))
            models = test_df['models']
            test_df = test_df.drop(columns='models')
            if metric_sum is None:
                metric_sum = test_df
            elif metric_sum.shape != test_df.shape:
                raise ValueError(f'difference metric shape on test {summ}:{test}')
            else:
                metric_sum = metric_sum.add(test_df, fill_value=0)
        metric_sum = metric_sum.div(len(summarization_list))
        metric_sum.insert(0, 'models', models, True)
        test_dir = join(output_dir, test)
        os.makedirs(test_dir, exist_ok=True)
        filepath = join(test_dir, 'metrics_average_comparison.csv')
        metric_sum.to_csv(filepath, index=False)
        


def average_tests_score(working_dir):
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


if __name__ == "__main__":
    summarize_model({"netkeys" : "1.0.1.0", 
                     "prufer": "1.0.6.0",
                     "kruskal": "1.0.2.0", 
                     "prim": "1.0.4.0",
                     "guided prim": "1.0.5.0"}, working_dir="results/_small/multi_hop")

