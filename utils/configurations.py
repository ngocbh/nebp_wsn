from yaml import Loader

import os
import yaml

def load_config(filepath, model):
    file = open(filepath, mode='r')
    full_config = yaml.load(file, Loader=Loader)
    args = list(map(int, model.split('.')))
    
    config = {}
    config['models'] = full_config['models'][args[0]]
    config['data'] = full_config['data'][args[1]]
    config['encoding'] = full_config['encoding'][args[2]]
    config['algorithm'] = full_config['algorithm'][args[3]]
    
    return config
    
def gen_output_dir(filename, model):
    output_dir = filename.replace('data', 'results')
    output_dir = os.path.dirname(output_dir)
    output_dir = os.path.join(output_dir, model)
    return output_dir

if __name__ == "__main__":
    working_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = '../configs/_configurations.yml'
    # load_config(os.path.join(working_dir, config_file), model="0.0.0.0")
    filename = 'data/small/multi_hop/ga-dem1_r25_1_0.json'
    gen_output_dir(filename, '0.0.0.0')
 

