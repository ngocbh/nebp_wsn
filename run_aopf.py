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
TESTING = False

def is_dominated(a, b):
    dominated = False
    for i in range(len(a)):
        if a[i] > b[i]:
            return False
        elif a[i] < b[i]:
            dominated = True
    return dominated

def nondominated_sort(population):

    fronts = list()
    num_dominators = dict()
    slaves = dict()

    # find fisrt pareto front
    fronts.append(list())
    for p1 in population:
        slaves[p1] = list()
        num_dominators[p1] = 0

        for p2 in population:
            if is_dominated(p1, p2):
                slaves[p1].append(p2)
            elif is_dominated(p2, p1):
                num_dominators[p1] += 1

        if num_dominators[p1] == 0:
            fronts[0].append(p1)

    # find other pareto front
    i = 0
    while len(fronts[i]) > 0:
        current_front = list()
        for p1 in fronts[i]:
            for p2 in slaves[p1]:
                num_dominators[p2] -= 1
                if num_dominators[p2] == 0:
                    current_front.append(p2)

        i += 1
        fronts.append(current_front)

    if len(fronts[-1]) == 0:
        fronts.pop()

    return fronts


def run_aopf():
    ept = 0 if TESTING else 1
    input_dir = './data/test' if TESTING else './data/ept_efficiency'

    output_dir = './results/test/referenced_pareto' if TESTING else './results/ept_efficiency/referenced_pareto'
    testset = 0 if TESTING else 3
    testnames = ['dem'] if TESTING else ['']
    k = 5 if TESTING else 100

    config = None
    if TESTING:
        config = {'models': {}, 'algorithm': {}}
        config['models']['gens'] = 2
        config['algorithm']['pop_size'] = 10
        config['algorithm']['selection_size'] = 10
    else:
        config = {'models': {}, 'algorithm': {}}
        config['models']['gens'] = 1000
        config['algorithm']['pop_size'] = 100
        config['algorithm']['selection_size'] = 1000

    solvers = [solver_mhn_gprim3, solver_mhn_kruskal, solver_mhn_nrk, solver_mhn_prim, solver_mhn_prufer]
    xs = [8, 2, 1, 4, 6]
    models = ['{}.{}.{}.0'.format(ept, testset, x) for x in xs]

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        print('working on file: ', file)
        if ('dem' not in file) or (testnames is not None and all(e not in file for e in testnames)):
            continue
        solution_pool = set()
        for i, solver in enumerate(solvers):
            results = joblib.Parallel(n_jobs=-1)(joblib.delayed(solver.solve)(
                os.path.join(input_dir, file), output_dir=None, model=models[i], 
                config=config, save_history=False, save=False, seed=seed) for seed in range(1, k+1))
            for result in results:
                temp = [tuple(x) for x in result.all_objectives()]
                solution_pool.update(temp)
        
        print(solution_pool)
        fronts = nondominated_sort(solution_pool)
        pareto_front = fronts[0]
        pareto_front.sort()
        print(pareto_front)
        with open(os.path.join(output_dir, file.replace('.json', '.txt')), mode='w') as f:
            for s in pareto_front:
                f.write('{} {}\n'.format(s[0], s[1]))
        


if __name__ == '__main__':
    run_aopf()


