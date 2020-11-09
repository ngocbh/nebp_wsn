import os
import pandas as pd
import re
from collections import defaultdict
from utils import WusnInput
from problems import MultiHopProblem

distribution = {'ga': 'Gaussian', 'no': "Gamma", 'uu': "Uniform"}

def get_graph_dense(inp, problem):
    a = len(problem._idx2edge)
    b = inp.num_of_relays + inp.num_of_relays * inp.num_of_sensors + inp.num_of_sensors * (inp.num_of_sensors-1)/2
    n = inp.num_of_sensors + inp.num_of_relays + 1
    b = n * (n-1) / 2
    return a / b 

def get_data_summarization():
    inp_dirs = ['./data/ept_efficiency', './data/ept_scalability']

    data = defaultdict(list)

    for inp_dir in inp_dirs:
        for file in os.listdir(inp_dir):
            if 'NIn' not in file:
                continue
            x = re.split('_|-|\.', file)
            inp = WusnInput.from_file(os.path.join(inp_dir, file))
            problem = MultiHopProblem(inp)
            data['set'].append('')
            data['instance'].append(x[0])
            data['type'].append(x[1])
            data['distribution'].append(distribution[x[2]])
            data['terrain'].append(x[3].replace('dem', 'T'))
            data['S'].append('${} \\times {}$'.format(inp.W, inp.H))
            data['N'].append(inp.num_of_relays)
            data['M'].append(inp.num_of_sensors)
            data['density'].append('{:.2f}'.format(get_graph_dense(inp, problem)))

    df = pd.DataFrame(data)
    df = df.sort_values(by='instance', key=lambda col : col.apply(lambda x : int(x.replace('NIn', ''))))
    df = df.reset_index(drop=True)
    df.at[0, 'set'] = '$T1$'
    df.at[9, 'set'] = '$T2$'
    df.to_csv('results/data_sum.csv', index=False)
            


if __name__ == '__main__':
    get_data_summarization()
