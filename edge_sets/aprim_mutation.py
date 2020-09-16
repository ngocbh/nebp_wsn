"""
File: aprim_mutation.py
Created by ngocjr7 on 2020-09-15 15:44
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

from __future__ import absolute_import

from geneticpython.core.individual import Individual
from geneticpython.core.operators import Mutation
from geneticpython.utils.validation import check_random_state
from geneticpython.models import Tree, RootedTree

from copy import deepcopy
from itertools import chain


class APrimMutation(Mutation):

    def mutate(self, indv: Individual, random_state=None):
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
        
        used_relays = []
        unused_relays = []
        for i in range(1, tree.n+1):
            if tree.num_childs[i] == 0:
                unused_relays.append(i)
            else:
                used_relays.append(i)
        slt_relay = random_state.choice(unused_relays)
        used_relays.append(slt_relay)

        used_relays_mask = [False] * tree.number_of_vertices
        for e in used_relays:
            used_relays_mask[e] = True

        max_energy = tree.calc_max_energy_consumption()
        edges = deepcopy(tree.edges)
        tree.build_mprim_tree(max_energy, used_relays_mask, edges, random_state)

        ret_indv.encode(tree)

        return ret_indv
        
