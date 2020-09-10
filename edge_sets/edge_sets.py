"""
File: edge_sets.py
Created by ngocjr7 on 2020-08-27 00:15
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""
from geneticpython.models.tree import EdgeSets, KruskalTree
from geneticpython.utils.validation import check_random_state
from typing import List, Tuple
import numpy as np

class KruskalEdgeSets(EdgeSets):
    def __init__(self, number_of_vertices, solution: KruskalTree = None, potential_edges : List[Tuple]=None):
        self.solution = solution or KruskalTree(number_of_vertices)
        if potential_edges is not None:
            self.potential_edges = set()
            for i in range(number_of_vertices):
                for j in range(i):
                    self.potential_edges.add((j,i))
            self.potential_edges = list(self.potential_edges)
        else: 
            self.potential_edges = set()
            for u, v in potential_edges:
                if (u, v) not in self.potential_edges and (v, u) not in self.potential_edges:
                    self.potential_edges.add( (u,v) )
            self.potential_edges = list(self.potential_edges)

        super(KruskalEdgeSets, self).__init__(number_of_vertices)

    def random_init(self, random_state=None):
        random_state = check_random_state(random_state)
        weight = random_state.random(len(self.potential_edges))
        order = np.argsort(-weight)
        self.solution.initialize()
        genes = np.empty(self.chromosome.length)
        for i in order:
            u, v = self.potential_edges[i]
            # if  

        self.solution.repair()
        # hhaahahahah

    def decode(self) -> KruskalTree:
        self.solution.initialize()
        _is_valid = True
        for i in range(0, self.chromosome.length, 2):
            u, v = self.chromosome[i, i+2]
            _is_valid &= self.solution.add_edge(u, v)

        self.solution._is_valid = _is_valid
        self.solution.repair()
        return self.solution

    def encode(self, solution : KruskalTree, random_state=None):
        order = [i for i in range(solution.number_of_vertices-1)]
        # random_state.shuffle(order)

        genes = [0] * self.chromosome.length
        edge_list = list(solution.edges)
        for i, j in enumerate(order):
            genes[2*i], genes[2*i+1] = edge_list[j]

        self.update_genes(genes) 

