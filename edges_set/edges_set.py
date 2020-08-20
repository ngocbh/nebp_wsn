from __future__ import absolute_import

from geneticpython.models.tree import EdgeSets, KruskalTree
from geneticpython.utils.validation import check_random_state

from problems import WusnProblem

from random import Random 

class WusnSolution(KruskalTree):
    def __init__(self, problem: WusnProblem):
        self.potential_edges = problem._edges
        self.n = problem._num_of_relays
        self.m = problem._num_of_sensors
        super(WusnSolution, self).__init__(self.n + self.m + 1, root=0)

class WusnEdgeSets(EdgeSets):
    def __init__(self, problem: WusnProblem):
        self.solution = WusnSolution(problem)
        super(WusnEdgeSets, self).__init__(problem.number_of_vertices)

    def decode(self) -> WusnSolution:
        self.solution.initialize()
        _is_valid = True
        for i in range(0, self.chromosome.length, 2):
            u, v = self.chromosome[i, i+2]
            _is_valid &= self.solution.add_edge(u, v)

        self.solution._is_valid = _is_valid
        return self.solution

    @classmethod
    def encode(cls, solution : KruskalTree, random_state=None):
        order = [i for i in range(solution.number_of_vertices)]
        random_state.shuffle(order)

        genes = []  

        
