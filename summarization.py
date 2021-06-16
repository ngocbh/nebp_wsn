"""
File: summarization.py
Created by ngocjr7 on 2020-09-08 10:51
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""
from geneticpython.tools.visualization import visualize_fronts
from geneticpython.tools.performance_indicators import delta_apostrophe, C_metric, \
    SP, ONVG, HV_2d, IGD, delta, GD
from yaml import Loader
from os.path import join
from collections import OrderedDict
from matplotlib.ticker import MaxNLocator
import json
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import re
import itertools
from decimal import Decimal

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
NORMALIZE = True

def toint(x):
    x = re.findall(r'\d+', x)
    x = tuple(int(e) for e in x)
    return x

def num2tex(n, p, mean=True):
    if mean:
        if pow(10, - p) < n and n < pow(10, p):
            r = '{:.3g}'.format(n)
        elif -pow(10, p) < n and n < -pow(10, -p):
            r = '{:.3g}'.format(n)
        elif -1e-10 < n and n < 1e-10:
            r = 0
        else:
            r = '{:.2e}'.format(n)
            r = r.replace('-0', '-')
    else:
        if pow(10, - p) < n and n < pow(10, p):
            r = '{:.2g}'.format(n)
        elif -pow(10, p) < n and n < -pow(10, -p):
            r = '{:.2g}'.format(n)
        elif -1e-10 < n and n < 1e-10:
            r = 0
        else:
            r = '{:.1e}'.format(n)
            r = r.replace('-0', '-')
    return r

def normalize_pareto_front(pareto, P):
    if pareto is None:
        return None
    ret = []
    for s1, s2 in pareto:
        n1 = (s1 - P[0][0] + 1) / (P[1][0] - P[0][0] + 1)
        n2 = (s2 - P[1][1]) / (P[0][1] - P[1][1])
        ret.append((n1, n2))
    # print(pareto)
    # print(P)
    # print(ret)
    return ret

def read_pareto(filepath):
    data = json.load(open(filepath, mode='r'))
    pareto = set()
    for solution in data:
        objectives = (solution['num_used_relays'],
                      solution['energy_consumption'] )
        pareto.add(objectives)
    pareto = list(pareto)
    # pareto = normalize_pareto_front(pareto, P)
    # pareto = read_pareto_history(filepath)
    return pareto

def read_pareto_history(filepath):
    data = json.load(open(filepath, mode='r'))
    history = []
    for g in data:
        pareto = set()
        for solution in g["pareto_front"]:
            pareto.add(tuple([solution[0], solution[1] ]))
        pareto = list(pareto)
        pareto.sort()
        history.append(pareto)
    return history

def read_time(filepath):
    t = None 
    with open(filepath, mode='r') as f:
        data = f.read()
        t = float(data[14:])
    return t

def visualize_test(pareto_dict, output_dir, show=True, **kwargs):
    def do_axis(ax):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.set_yscale('logit')
        ax.grid(b=True, axis='y')
        ax.grid(b=False, axis='x')
        # for axis in ['top','bottom','left','right']:
        #     ax.spines[axis].set_linewidth(2)

    def do_plt(plt):
        # plt.rcParams.update({'font.size': 20})
        # plt.rcParams.update({'lines.linewidth': 2})
        # plt.rcParams.update({'axes.linewidth': 2})
        plt.style.use('seaborn-white')
        plt.grid(False)

    filepath = os.path.join(output_dir, 'front_comparison.png')
    visualize_fronts(pareto_dict,
                     filepath=filepath,
                     title='',
                     objective_name=['No. selected relays', 'Energy consumption'],
                     save=True, show=show, do_axis=do_axis, do_plt=do_plt, dpi=400, frameon=True, **kwargs)

def visualize_igd_over_generations(history_dict, output_dir, P, extreme_points, marker=None,
                                   linewidth=2, markersize=5, linestyle='--', fillstyle=None, **kwargs):
    def normalize_pareto_front_1(S, P):
        S.sort()
        ret = [S[0]]
        for i in range(1, len(S)):
            while ret[-1][0] < S[i][0] - 1:
                x = ret[-1]
                ret.append((x[0] + 1, x[1]))
            ret.append(S[i])
        while ret[-1][0] < np.max(P[:, 0]):
            ret.append((ret[-1][0] + 1, ret[-1][1]))

        return ret

    if NORMALIZE:
        normalized_optimal_pareto = normalize_pareto_front(P, extreme_points)
    else:
        normalized_optimal_pareto = P
    data = {}
    for name, history in history_dict.items():
        igds = []
        # print(name)
        Ss = set()
        for i, S in enumerate(history):
            # if i == 0:
                # print(S)
            if NORMALIZE:
                # S = normalize_pareto_front_1(S, P)
                normalized_S = normalize_pareto_front(S, extreme_points)
            else:
                normalized_S = S
            Ss.update(normalized_S)
            Ss_list = list(Ss)
            Ss_list.sort()
            # if i == 0:
            # print(Ss_list)
            # print(S)
            # print(normalized_S)
            # print(normalized_optimal_pareto)
            igds.append(IGD(Ss_list, normalized_optimal_pareto))
            # print(igds)
            # raise ValueError
        # print(name)
        # print(igds)
        data[name] = igds

    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'lines.linewidth': 3})
    plt.rcParams.update({'axes.linewidth': 2})
    plt.style.use('seaborn-white')
    plt.grid(True)
    fig, ax = plt.subplots()

    # exit()


    marker = marker or ['+', 'v', '^', 'o', (5, 1), (5, 0)]
    marker.reverse()
    iter_marker = itertools.cycle(marker)

    if fillstyle == 'flicker':
        fillstyle = ['full', 'none']
    elif fillstyle == 'all':
        fillstyle = ['none', 'top', 'bottom', 'right', 'left', 'full']
    else:
        fillstyle = [fillstyle]
    iter_fillstyle = itertools.cycle(fillstyle)

    num = 0
    palette = plt.get_cmap('Set1')

    for name, history in data.items():
        num += 1
        m = next(iter_marker)
        fs = next(iter_fillstyle)
        ax.plot(list(range(len(history))), history, label=name,
                linewidth=3, linestyle='-', color=None, alpha=0.9, fillstyle=fs)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(b=True, axis='y')
    ax.grid(b=False, axis='x')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    # ax.set_yscale('log')
    filepath = os.path.join(output_dir, 'igd_over_generations.png')
    plt.ylabel("$IDG$")
    plt.xlabel("No. generations")

    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=3, fontsize=15)
    plt.tight_layout()
    # plt.title("IGD over generations")
    plt.savefig(filepath, dpi=400)
    plt.close('all')


def summarize_metrics(pareto_dict, time_dict, output_dir, r, referenced=False, P=None, Pe=None):
    metrics = {}
    metrics['models'] = list(pareto_dict.keys())
    if referenced:
        metrics['igd'] = []
    metrics['delta'] = []
    metrics['spacing'] = []
    metrics['onvg'] = []
    metrics['hypervolume'] = []
    metrics['time'] = []
    for key in pareto_dict.keys():
        metrics['c_' + key] = []
    metrics['score'] = []

    n = len(pareto_dict.keys())
    c_matrix = [[] for _ in range(n)]
    i = 0

    for name, pareto in pareto_dict.items():
        # print(name)
        # print(pareto)
        if name != metrics['models'][i]:
            raise ValueError("Summarize metrics error")
        # print(pareto)
        if NORMALIZE:
            normalized_pareto = normalize_pareto_front(pareto, Pe)
            normalized_P = normalize_pareto_front(P, Pe)
            delta_P = [(0, 1), (1, 0)]
            hyper_r = (1,1)
        else:
            normalized_pareto = pareto
            normalized_P = P
            delta_P = Pe
            hyper_r = r
        # print(pareto)

        # print(pareto, normalized_pareto)
        # print(normalized_pareto)
        # print(normalized_P)
        if referenced:
            metrics['igd'].append(IGD(normalized_pareto, normalized_P))
            # print(metrics['igd'])
        # raise ValueError

        # print(name)
        if Pe is None:
            metrics['delta'].append(delta_apostrophe(normalized_pareto))
            raise ValueError()
        else:
            metrics['delta'].append(delta(normalized_pareto, delta_P))

        metrics['time'].append(time_dict[name])
        metrics['spacing'].append(SP(normalized_pareto))
        metrics['onvg'].append(ONVG(normalized_pareto))

        # print(Pe)
        # print('hyper_r', hyper_r)
        # print(pareto)
        # print(normalized_pareto)
        # print(HV_2d(normalized_pareto, hyper_r))
        # print('\n\n')
        metrics['hypervolume'].append(HV_2d(normalized_pareto, hyper_r))

        for other_name, other_pareto in pareto_dict.items():
            c = C_metric(pareto, other_pareto)
            metrics['c_' + other_name].append(c)
            c_matrix[i].append(c)

        i += 1
    # exit()

    for i in range(n):
        score = 0
        for j in range(n):
            if c_matrix[i][j] - c_matrix[j][i] > 0:
                score += 1
        metrics['score'].append(score)

    df = pd.DataFrame(data=metrics)
    filepath = os.path.join(output_dir, 'metrics_comparison.csv')
    df.to_csv(filepath, index=False)


def summarize_test(testname, model_dict, working_dir, cname, referenced=False, referenced_dir=None, **kwargs):
    # print(testname)
    absworking_dir = os.path.join(WORKING_DIR, working_dir)
    pareto_dict = {}
    config_dict = {}
    history_dict = {}
    time_dict = {}
    rs = []
    Ps = []
    for name, model in model_dict.items():
        model_dir = os.path.join(absworking_dir, model)
        test_dir = os.path.join(model_dir, testname)
        if not os.path.isdir(test_dir) and os.path.isfile(os.path.join(test_dir, 'done.flag')):
            continue

        if os.path.isfile(os.path.join(test_dir, 'P.txt')):
            P = np.loadtxt(os.path.join(test_dir, 'P.txt')) 
            Ps.append(P)

        pareto = read_pareto(os.path.join(test_dir, 'pareto-front.json'))
        history = read_pareto_history(os.path.join(test_dir, 'history.json'))
        pareto_dict[name] = pareto
        time_dict[name] = read_time(os.path.join(test_dir, 'time.txt'))

        if os.path.isfile(os.path.join(test_dir, 'r.txt')):
            with open(os.path.join(test_dir, 'r.txt')) as f:
                data = f.read().split()
                r = tuple([float(e) for e in data])
                rs.append(r)

        config = yaml.load(
            open(os.path.join(test_dir, '_config.yml')), Loader=Loader)
        config["model_name"] = model
        config_dict[name] = config

        if referenced:
            history = read_pareto_history(os.path.join(test_dir, 'history.json'))
            history_dict[name] = history

    for r in rs:
        if r != rs[0]:
            raise ValueError("Catched different r {}".format(testname))

    for Pe in Ps:
        if np.any(Pe != Ps[0]):
            raise ValueError("Catched different P {}".format(testname))

    r = rs[0] if len(rs) > 0 else (200,1)
    Pe = Ps[0] if len(Ps) > 0 else np.array([[1, 0.02], [100, 0.0001]]) 

    output_dir = os.path.join(absworking_dir, cname)
    out_test_dir = os.path.join(output_dir, testname)
    os.makedirs(out_test_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config_comparison.yml'), mode='w') as f:
        f.write(yaml.dump(config_dict))

    P = None
    if referenced:
        referenced_file = os.path.join(referenced_dir, testname + '.txt')
        P = np.loadtxt(referenced_file)

    summarize_metrics(pareto_dict, time_dict, output_dir=out_test_dir, r=r, referenced=referenced, P=P, Pe=Pe)
    if referenced:
        visualize_test(pareto_dict, output_dir=out_test_dir, show=False, referenced_points=P, **kwargs)
        visualize_igd_over_generations(history_dict, output_dir=out_test_dir, P=P, fillstyle='flicker', extreme_points=Pe)
    else:
        visualize_test(pareto_dict, output_dir=out_test_dir, show=False, **kwargs)

def combine_figure(outfile, working_dir, test_list, model_dict):
    test_pattern = ['NIn']
    used_tests = [e for e in test_list if any(x in e for x in test_pattern)]
    print(used_tests)
    plt.style.use('seaborn-white')
    
    def plot_test(ax, working_dir, testname, model_dict):
        pareto_dict = {}


        marker = ['+', 'v', '^', 'o', (5, 1), (5, 0)]
        marker.reverse()
        iter_marker = itertools.cycle(marker)
        fillstyle = ['none']
        iter_fillstyle = itertools.cycle(fillstyle)
        for name, model in model_dict.items():
            model_dir = os.path.join(working_dir, model)
            test_dir = os.path.join(model_dir, testname)
            if not os.path.isdir(test_dir) and os.path.isfile(os.path.join(test_dir, 'done.flag')):
                continue

            pareto = read_pareto(os.path.join(test_dir, 'pareto-front.json'))
            pareto.sort()
            pareto_dict[name] = pareto
            obj1 = [e[0] for e in pareto]
            obj2 = [e[1] for e in pareto]
            ax.plot(obj1, obj2, marker=next(iter_marker), linestyle='--', linewidth=1, 
                    markersize=3, fillstyle=next(iter_fillstyle), alpha=0.8)
        ax.set_title(testname.split('_')[0])

    no_col = 4
    no_tests = len(used_tests)
    no_row = (no_tests + no_col - 1)//no_col

    fig, axs = plt.subplots(no_row, no_col, figsize=(9, 5), dpi=100)
    plt.grid(False)

    for i, testname in enumerate(used_tests):
        print(i, testname)
        idx, idy = i // no_col, i % no_col
        plot_test(axs[idx, idy], working_dir, testname, model_dict)


    ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("No. selected relays")
    plt.ylabel("Energy consumption")

    marker = ['+', 'v', '^', 'o', (5, 1), (5, 0)]
    marker.reverse()
    iter_marker = itertools.cycle(marker)
    fillstyle = ['none']
    iter_fillstyle = itertools.cycle(fillstyle)
    names = ['HMOEA', 'Prufer', 'NetKeys', 'Prim', 'Kruskal',  'GPrim']
    for name in names:
        ax.plot([], linestyle='--', marker=next(iter_marker), fillstyle=next(iter_fillstyle), label=name, alpha=0.8)

    ax.legend(loc='upper center', bbox_to_anchor=(0., 1.06, 1., .106),
              ncol=6, frameon=True)
    fig.tight_layout()
    plt.savefig(outfile, dpi=400, bbox_inches='tight')
    # plt.show()
    plt.close('all')

def summarize_model(model_dict, working_dir, cname=None, testnames=None,
                    marker=None, markersize=20, plot_line=True, linewidth=0.8, linestyle='dashed',
                    referenced=False, referenced_dir=None, combined=False, **kwargs):

    print("Summarizing {}: {}".format(cname, model_dict))
    tests = set()
    absworking_dir = os.path.join(WORKING_DIR, working_dir)

    comparison_name = "sum"
    for model in model_dict.values():
        comparison_name += '+' + model
        model_dir = os.path.join(absworking_dir, model)
        for filename in os.listdir(model_dir):
            inp_name = f'{filename}.json'
            if ('dem' not in filename) or (testnames is not None and all(e not in inp_name for e in testnames)):
                continue
            tests.add(filename)

    cname = cname or comparison_name
    output_dir = os.path.join(absworking_dir, cname)
    os.makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, 'model_dict.json'), mode='w') as f:
        f.write(json.dumps(model_dict, indent=4))

    marker = ['+', (5, 1), 'v', '^', 'o', (5, 0)]
    marker.reverse()

    test_list = list(tests)
    try:
        test_list.sort(key=toint)
    except:
        print("ERROR")
        pass

    for test in test_list:
        summarize_test(test, model_dict, working_dir, cname,
                       markersize=7,
                       marker=marker,
                       plot_line=True,
                       linewidth=1.3,
                       linestyle='dashed',
                       fillstyle='none',
                       referenced=referenced,
                       referenced_dir=referenced_dir,
                       **kwargs)

    if combined:
        outfile = os.path.join(output_dir, 'combined_pareto_fronts.png')
        combine_figure(outfile, working_dir, test_list, model_dict)


def calc_average_metrics(summarization_list, working_dir, cname, testnames=None, referenced=False, bold=True, brief_name='average'):
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def plot_bar_chart(test_dir, data):
        plt.rcParams.update({'font.size': 17})
        plt.rcParams.update({'lines.linewidth': 3})
        plt.rcParams.update({'axes.linewidth': 2})
        plt.style.use('seaborn-white')
        plt.grid(False)
        palette = plt.get_cmap('Set1')
        PI_MAP = { 'igd': '$IGD$', 'delta': '$\Delta$', 'onvg': '$ONVG$', 'hypervolume': '$HV$', 'spacing': '$spacing$'}

        models = data.keys()
        for value in data.values():
            pis = list(value.keys())
            break
        if 'igd' in pis:
            pis.remove('igd')

        if 'time' in pis:
            pis.remove('time')

        fig, host = plt.subplots()
        par1 = host.twinx()
        if len(pis) == 3:
            fig.subplots_adjust(right=0.75)
            par2 = host.twinx()
            par2.spines["right"].set_position(("axes", 1.2))
            make_patch_spines_invisible(par2)
            par2.spines["right"].set_visible(True)
            axs = [host, par1, par2]
        else:
            axs = [host, par1]

        ind = np.arange(len(models))
        model_width = 0.75

        idx = 0
        npis = len(pis)
        pi_width = model_width / npis
        bars = []

        for pi in pis:
            pi_mean = []
            pi_std = []
            lower=False
            for model in models:
                x = np.array(data[model][pi])
                x_mean = np.mean(x)
                x_std = np.std(x)
                if x_mean - x_std < 0:
                    lower=True
                pi_mean.append(x_mean)
                pi_std.append(x_std)

            px = axs[idx].bar(ind + (idx + 0.5) * pi_width - model_width/2, pi_mean, pi_width,
                   bottom=0, edgecolor = 'black',color=palette(idx+1), alpha=0.9, yerr=pi_std, capsize=7, label=PI_MAP[pi])
            bars.append(px)
            if lower:
                axs[idx].set_ylim((0, None))
            idx += 1

        for i, ax in enumerate(axs):
            color = bars[i].patches[0].get_facecolor()
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)
            ax.set_ylabel(PI_MAP[pis[i]])
            ax.grid(False)

        host.set_xticks(ind)
        host.tick_params(axis='x', labelrotation=40)
        ax.set_xticklabels(models, fontsize=14)
        # ax.set_ylabel()
        # ax.set_title("")
        # axs[-1].legend(bars, [PI_MAP[pi] for pi in pis], frameon=True)
        fig.tight_layout()
        filepath = os.path.join(test_dir, 'bar_plot.png')
        plt.savefig(filepath, dpi=400)
        plt.close('all')

    def plot_box_chart(out_dir, data, brief_name=None):
        plt.style.use('seaborn-white')
        def box_plot(ax, data):
            n = len(data)
            width= 1.1 / (n + (n+1)/2)
            pos = [ -0.05 + (1 + 3*i/2)*width for i in range(n)]
            flierprops = dict(marker='+', markersize=3)
            meanprops = dict(linewidth=3)
            medianprops = dict(linewidth=2)
            ax.boxplot(data, positions=pos, widths=width, notch=False,
                       sym='+', meanline=True, showmeans=True, 
                       meanprops=meanprops, medianprops=medianprops,
                       flierprops=flierprops)

        def name_plot(ax, name):
            ax.text(0.5, 0.5, name.upper(), ha='center', va='center', weight="bold", size=15)
            ax.grid(False)

        models = list(data[0].keys())
        n = len(models)
        fig, axs = plt.subplots(n, n, figsize=(9, 9), sharey=True, sharex=True, dpi=100)
        plt.grid(True)
        for i in range(n):
            for j in range(n):
                if i == j:
                    name_plot(axs[i, j], models[i])
                else:
                    sub_data = []
                    for test in data:
                        sub_data.append(test[models[i]][models[j]])
                    box_plot(axs[i, j], sub_data)
                    axs[i, j].grid(b=True, axis='y')
                    axs[i, j].grid(b=False, axis='x')


        for ax in axs.flat:
            ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), aspect=1)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
            ax.set_yticks([i*0.2 for i in range(6)])        

        fig.tight_layout(pad=1)

        filepath = os.path.join(out_dir, '{}_c_metric.png'.format(brief_name))
        plt.savefig(filepath, dpi=400)
        # plt.show()
        plt.close('all')

    def compact_data_to_csv(output_dir, data, brief_name=None, bold=True):
        def normalize_name(name):
            x = re.split(r'_|-|\.', name)
            return x[0] if 'NIn' in name else name
        models = list(next(iter(data.values())).keys())
        pis = list(next(iter(next(iter(data.values())).values())).keys())
        bold_pis = {'hypervolume': 1, 'igd': -1, 'delta': -1, 'onvg': 1, 'time': -1}
        brief_name = brief_name or 'results'


        for pi in pis:
            normalized_data = {}
            normalized_data['Instance'] = []
            raw_data = {}
            raw_data['Instance'] = []

            for model in models:
                normalized_data[f'{model}-mean'] = []
                if pi != 'time':
                    normalized_data[f'{model}-std'] = []
                raw_data[f'{model}-mean'] = []
                if pi != 'time':
                    raw_data[f'{model}-std'] = []


            for testname, test_data in data.items():
                normalized_data['Instance'].append(normalize_name(testname))
                raw_data['Instance'].append(normalize_name(testname))
                print(testname)
                best = np.max(np.array([ np.mean(np.array(x[pi]) * bold_pis[pi]) for x in test_data.values() ]))

                for model, model_data in test_data.items():
                    d = np.array(model_data[pi])
                    d_mean = np.mean(d)
                    d_std = np.std(d)
                    if np.abs(d_mean*bold_pis[pi] - best) < 1e-10 and bold:
                        if pi == 'time':
                            d_mean_str = '\\textbf{' + '{:.1f}'.format(d_mean) + '}'
                        else:
                            d_mean_str = '\\textbf{' + str(num2tex(d_mean, 2)) + '}'
                            d_std_str = '$\pm$' + str(num2tex(d_std, 2, mean=False))
                    else:
                        if pi == 'time':
                            d_mean_str = '{:.1f}'.format(d_mean)
                        else:
                            d_mean_str = str(num2tex(d_mean, 2))
                            d_std_str =  '$\pm$' + str(num2tex(d_std, 2, mean=False))
                    normalized_data[f'{model}-mean'].append(d_mean_str)
                    raw_data[f'{model}-mean'].append(d_mean)
                    if pi != 'time':
                        normalized_data[f'{model}-std'].append(d_std_str)
                        raw_data[f'{model}-std'].append(d_std)

            filepath = os.path.join(output_dir, '{}_{}.csv'.format(brief_name, pi))
            df = pd.DataFrame(data=normalized_data)
            df = df.sort_values(by='Instance', key= lambda col : col.apply(toint))
            df.to_csv(filepath, index=False)

            filepath = os.path.join(output_dir, 'raw_{}_{}.csv'.format(brief_name, pi))
            df = pd.DataFrame(data=raw_data)
            df = df.sort_values(by='Instance', key= lambda col: col.apply(toint))
            df.to_csv(filepath, index=False)

    all_tests = set()
    absworking_dir = os.path.join(WORKING_DIR, working_dir)
    for model in summarization_list:
        model_dir = os.path.join(absworking_dir, model)
        for filename in os.listdir(model_dir):
            inp_name = f'{filename}.json'
            if ('dem' not in filename) or (testnames is not None and all(e not in inp_name for e in testnames)):
                continue
            all_tests.add(filename)

    output_dir = os.path.join(absworking_dir, cname)
    os.makedirs(output_dir, exist_ok=True)

    feasible_tests = all_tests.copy()
    for model in summarization_list:
        model_dir = os.path.join(absworking_dir, model)
        model_tests = set()
        for filename in os.listdir(model_dir):
            inp_name = f'{filename}.json'
            if ('dem' not in filename) or (testnames is not None and all(e not in inp_name for e in testnames)):
                continue
            model_tests.add(filename)
        feasible_tests.intersection(model_tests)

    with open(os.path.join(output_dir, 'status.txt'), mode='w') as f:
        f.write('Feasible_tests: ')
        f.write(str(feasible_tests))
        f.write('\nUnsolved_tests: ' +
                str(all_tests.difference(feasible_tests)))

    boxchart_data = []
    barchart_data = {}
    feasible_tests = list(feasible_tests)
    feasible_tests.sort(key=toint)
    print(feasible_tests)

    for test in feasible_tests:
        metric_sum = None
        models = None
        barchart_test_data = None
        boxchart_test_data = None
        if referenced:
            bar_metric_temp = OrderedDict({ 'igd': [], 'onvg': [], 'delta': [], 'hypervolume': [], 'time': []})
        else:
            bar_metric_temp = OrderedDict({ 'onvg': [], 'delta': [], 'hypervolume': [], 'time': []})

        for summ in summarization_list:
            test_dir = join(join(absworking_dir, summ), test)
            test_df = pd.read_csv(join(test_dir, 'metrics_comparison.csv'))
            models = test_df['models']

            if boxchart_test_data is None or barchart_test_data is None:
                boxchart_test_data = {}
                barchart_test_data = {}
                for model in models:
                    barchart_test_data[model] = copy.deepcopy(bar_metric_temp)
                    boxchart_test_data[model] = {}
                    for other_model in models:
                        boxchart_test_data[model][other_model] = []

            for _, row in test_df.iterrows():
                model = row['models']
                for key in bar_metric_temp.keys():
                    barchart_test_data[model][key].append(row[key])

                for other_model in models:
                    boxchart_test_data[model][other_model].append(
                        row[f'c_{other_model}'])

            test_df = test_df.drop(columns='models')
            if metric_sum is None:
                metric_sum = test_df
            elif metric_sum.shape != test_df.shape:
                raise ValueError(
                    f'difference metric shape on test {summ}:{test}')
            else:
                metric_sum = metric_sum.add(test_df, fill_value=0)

        metric_sum = metric_sum.div(len(summarization_list))
        metric_sum.insert(0, 'models', models, True)
        test_dir = join(output_dir, test)
        os.makedirs(test_dir, exist_ok=True)
        filepath = join(test_dir, 'metrics_average_comparison.csv')
        metric_sum.to_csv(filepath, index=False)

        # print(barchart_data)
        plot_bar_chart(test_dir, barchart_test_data)
        boxchart_data.append(boxchart_test_data)
        barchart_data[test] = barchart_test_data

    plot_box_chart(output_dir, boxchart_data, brief_name=brief_name)
    compact_data_to_csv(output_dir, barchart_data, brief_name=brief_name, bold=bold)

def average_tests_score(working_dir):
    metric_sum = None
    n_tests = 0
    for test in os.listdir(working_dir):
        if 'dem' in test:
            test_dir = os.path.join(working_dir, test)
            test_df = pd.read_csv(join(test_dir, 'metrics_average_comparison.csv'))
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
    marker = ['>', (5, 0), (5, 1), (5, 2), '+', 'o']
    marker.reverse()
    summarize_model({"netkeys": "1.7.1.0",
                     "prufer": "1.7.6.0",
                     "kruskal": "1.7.2.0",
                     "prim": "1.7.4.0",
                     "hmoea": "1.7.7.0",
                     "gprim": "1.7.8.0"},
                    working_dir="results/_tiny/multi_hop",
                    s=20,
                    marker=marker,
                    plot_line=True,
                    linewidth=0.8,
                    linestyle='dashed')
