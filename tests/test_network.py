
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from problems import WusnProblem, MultiHopProblem
from rooted_networks import MultiHopNetwork
from utils import WusnInput
from utils import WusnConstants
from utils.point import distance

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

def test(filename, chromosome):
    wusnfile = os.path.join(WORKING_DIR, filename)
    inp = WusnInput.from_file(wusnfile)
    problem = MultiHopProblem(inp, 8)

    edges = []
    for i in range(0,len(chromosome), 2):
        edges.append((chromosome[i], chromosome[i+1]))

    network = MultiHopNetwork(problem) 
    network.from_edge_list(edges)
    print(network.parent)
    print(network.num_childs)
    print(network.calc_max_energy_consumption())
    
        

if __name__ == '__main__':
    chromosome = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0, 32, 0, 33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 40, 60, 66, 89, 110, 95, 115, 94, 100, 90, 104, 20, 44, 36, 65, 93, 119, 10, 48, 85, 109, 52, 73, 86, 94, 9, 62, 102, 110, 31, 49, 101, 110, 22, 79, 97, 118, 38, 57, 17, 56, 100, 110, 99, 116, 106, 113, 98, 117, 14, 78, 81, 92, 92, 93, 92, 103, 29, 73, 84, 112, 56, 75, 49, 81, 84, 116, 44, 70, 11, 68, 102, 104, 88, 113, 96, 113, 71, 74, 69, 80, 42, 70, 87, 110, 90, 107, 46, 61, 43, 44, 49, 90, 49, 51, 51, 114, 58, 63, 88, 92, 7, 47, 105, 120, 6, 77, 28, 76, 93, 98, 91, 103, 61, 78, 82, 107, 24, 69, 24, 63, 92, 112, 83, 94, 111, 117, 92, 95, 15, 50, 97, 101, 61, 64, 40, 53, 41, 62, 4, 45, 6, 60, 5, 54, 40, 59, 82, 108, 72, 74, 13, 67, 10, 55, 105, 110, 56, 74, 87, 109]
    test('../data/small/multi_hop/ga-dem1_r25_1_40.json', chromosome)
