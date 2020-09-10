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

    i = 0
    for name, pareto in pareto_dict.items():
        if name != metrics['models'][i]:
            raise ValueError("Summarize metrics error")
        i += 1

        metrics['delta'].append(delta_metric(pareto))
        metrics['spacing'].append(spacing_metric(pareto))
        metrics['nds'].append(non_dominated_solutions(pareto))
        for other_name, other_pareto in pareto_dict.items():
            metrics['c_' + other_name].append(coverage_metric(pareto, other_pareto))

    df = pd.DataFrame(data=metrics)
    filepath = os.path.join(output_dir, 'metrics_comparison.csv')
    df.to_csv(filepath, index=False)

def summarize_test(testname, model_dict, working_dir):
    absworking_dir = os.path.join(WORKING_DIR, working_dir)
    pareto_dict = {}
    config_dict = {}
    comparison_name = "sum"
    for name, model in model_dict.items():
        model_dir = os.path.join(absworking_dir, model)
        test_dir = os.path.join(model_dir, testname)
        pareto = read_pareto(os.path.join(test_dir, 'pareto-front.json'))
        pareto_dict[name] = pareto
        config = yaml.load(open(os.path.join(test_dir, '_config.yml')), Loader=Loader)
        config["model_name"] = model
        config_dict[name] = config
        comparison_name += '+' + model
    
    output_dir = os.path.join(absworking_dir, comparison_name)
    out_test_dir = os.path.join(output_dir, testname)
    os.makedirs(out_test_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config_comparison.yml'), mode='w') as f:
        f.write(yaml.dump(config_dict))

    visualize_test(pareto_dict, output_dir=out_test_dir, show=False)
    summarize_metrics(pareto_dict, output_dir=out_test_dir)

def summarize_model(model_dict, working_dir):
    tests = set()
    absworking_dir = os.path.join(WORKING_DIR, working_dir)
    
    for model in model_dict.values():
        model_dir = os.path.join(absworking_dir, model)
        for filename in os.listdir(model_dir):
            if 'dem' in filename:
                tests.add(filename)

    for test in tests:
        summarize_test(test, model_dict, working_dir)



if __name__ == "__main__":
    summarize_model({"netkeys-old" : "1.0.1.0.1", "kruskal-old" : "1.0.2.0.1",
                     "netkeys-new" : "1.0.1.0", "kruskal-new": "1.0.2.0.2", "kruskal-nnew": "1.0.2.0"}, working_dir="results/small/multi_hop")
