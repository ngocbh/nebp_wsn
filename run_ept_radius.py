
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
import itertools
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
TESTING=False


def run_ept():
    ept = 0 if TESTING else 1
    input_dir = './data/test' if TESTING else './data/ept_radius'
    output_dir = None
    testset = 0 if TESTING else 9
    testnames = ['dem'] if TESTING else ['']
    k = 10
    config = None
    if TESTING:
        config = {'models': {}, 'algorithm': {}}
        config['models']['gens'] = 2
        config['algorithm']['pop_size'] = 10
        config['algorithm']['selection_size'] = 10

    run.run_mhn_experiment(ept, input_dir, output_dir, testset, testnames, k, overwrite=RERUN, config=config, brief_name='s4', combined=True)

def plot_ept():
    wdir = 'results/ept_radius/average1-10'
    
    def plot(file, name):
        df = pd.read_csv(file)
        basefile = os.path.splitext(file)[0]
        cols = filter(lambda x: 'mean' in x, list(df.columns))
        cols = list(cols)
        legends = [col[:-5] for col in cols]
        df['ins'] = df['Instance'].apply(lambda x: x.split('_')[0])
        df['idx'] = df['Instance'].apply(lambda x: x.split('_')[1][1:])
        inses = df['ins'].unique()
        for ins in inses:
            ins_df = df.loc[df['ins'] == ins]
            ins_df = ins_df[['idx'] + cols]
            print(ins_df)
            plt.style.use('seaborn-white')
            plt.figure()
            plt.rcParams.update({'font.size': 20})
            plt.rcParams.update({'lines.linewidth': 3})
            plt.rcParams.update({'axes.linewidth': 2})

            ax = plt.figure().gca()

            marker = ['+', 'v', '^', 'o', (5, 1), (5, 0)]
            marker.reverse()
            iter_marker = itertools.cycle(marker)
            fillstyle = ['full', 'none']
            iter_fillstyle = itertools.cycle(fillstyle)
            ps = []
            for col in cols:
                p = ax.plot(ins_df['idx'], ins_df[col], label=col[:-5], marker=next(iter_marker),
                        linestyle=None, markersize=10)
                ps.append(p[0])


            ax.grid(b=True, axis='y')
            ax.grid(b=False, axis='x')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel(name)
            ax.set_xlabel("Radius")

            if 'Hyper' in name:
                plt.legend(ps[:2], legends[:2], frameon=False, 
                          loc='upper center', bbox_to_anchor=(0., 1.1, 1., .11), ncol=2)
            elif 'Delta' in name:
                plt.legend(ps[2:4], legends[2:4], frameon=False, 
                          loc='upper center', bbox_to_anchor=(0., 1.1, 1., .11), ncol=2)
            elif 'ONVG' in name:
                plt.legend(ps[4:], legends[4:], frameon=False, 
                          loc='upper center', bbox_to_anchor=(0., 1.1, 1., .11), ncol=2)


            out_filepath = join(wdir, ins + '_' + name + '.png')
            plt.tight_layout()
            # fig = plt.gcf()
            # fig.set_size_inches(4, 4)
            plt.savefig(out_filepath, dpi=400)
            plt.close('all')
        # print(df)

    plot(os.path.join(wdir, 'raw_s4_hypervolume.csv'), 'Hypervolume')
    plot(os.path.join(wdir, 'raw_s4_onvg.csv'), '$ONVG$')
    plot(os.path.join(wdir, 'raw_s4_delta.csv'), '$\Delta$')
    pass

if __name__ == '__main__':
    run_ept()
    plot_ept()

