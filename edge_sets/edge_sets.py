from __future__ import absolute_import

from geneticpython.models.tree import EdgeSets, KruskalTree
from geneticpython.utils.validation import check_random_state

from problems import WusnProblem
from networks import WusnNetwork

from random import Random 

class WusnEdgeSets(EdgeSets):
    def __init__(self, problem: WusnProblem, network: WusnNetwork):
        self.solution = network
        self.problem = problem
        super(WusnEdgeSets, self).__init__(problem.number_of_vertices)

    def decode(self) -> WusnNetwork:
        self.solution.initialize()
        _is_valid = True
        for i in range(0, self.chromosome.length, 2):
            u, v = self.chromosome[i, i+2]
            _is_valid &= self.solution.add_edge(u, v)

        self.solution._is_valid = _is_valid
        self.solution.repair()
        return self.solution

    def encode(self, solution : WusnNetwork, random_state=None):
        order = [i for i in range(solution.number_of_vertices-1)]
        random_state.shuffle(order)

        genes = [0] * self.chromosome.length
        edge_list = list(solution.edges)
        for i, j in enumerate(order):
            genes[2*i], genes[2*i+1] = edge_list[j]

        self.update_genes(genes) 
            


        
