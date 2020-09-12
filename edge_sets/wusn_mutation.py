"""
File: wusn_mutation.py
Created by ngocjr7 on 2020-08-21 18:42
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

from __future__ import absolute_import

from geneticpython.core.individual import Individual
from geneticpython.core.operators import TreeMutation
from geneticpython.utils.validation import check_random_state
from geneticpython.models import Tree, RootedTree

from copy import deepcopy
from itertools import chain

class WusnMutation(TreeMutation):

    def mutate(self, indv: Individual, random_state=None):
        # make sure random_state is not None
        random_state = check_random_state(random_state)

        ret_indv = indv.clone()
        
        # if not do mutation, return the cloner of indv
        if random_state.random() >= self.pm:
            return ret_indv

        # decode individual to get a tree representation
        tree = ret_indv.decode()

        # make sure it is an instance of Tree
        if not isinstance(tree, Tree):
            raise ValueError(f"TreeMutation is only used on the individual\
                             that decode to an instance of Tree,\
                             got {type(tree)}")

        # copy the edges of previous tree
        edges = deepcopy(tree.edges)

        # find unused edges in tree, used to choose new edge
        unused_edges = list()
        for u, v in self.potential_edges: 
            if (u, v) not in tree.edges and (v, u) not in tree.edges:
                unused_edges.append((u, v))

        # choose new edge
        idx = random_state.random_integers(0, len(unused_edges)-1)
        new_edge = unused_edges[idx]

        # find cycle path after create new edge
        path = tree.find_path(source=new_edge[0], destination=new_edge[1])

        removable_edges = []
        for i in range(len(path)-1):
            if path[i] != 0 and path[i+1] != 0:
                removable_edges.append((path[i],path[i+1]))

        # choose random edge on cycle to remove (break cycle)
        idx = random_state.random_integers(0, len(removable_edges)-1)
        removed_edge = removable_edges[idx]

        # print(removed_edge, new_edge)
        if removed_edge in edges:
            edges.remove(removed_edge)
        else:
            edges.remove((removed_edge[1], removed_edge[0]))

        edges.append(new_edge)
        if isinstance(tree, RootedTree):
            edges = tree.sort_by_bfs_order(edges)

        # make new tree from edges
        tree.initialize()
        for u, v in edges:
            tree.add_edge(u, v)
        tree.repair()

        # print(tree.edges)
        # print(len(tree.edges), ret_indv.chromosome.length)
        # reencode tree to ret_indv
        try:
            ret_indv.encode(tree)
        except:
            raise ValueError("Cannot call encode method. TreeMutation requires encode method in Individual")

        return ret_indv
