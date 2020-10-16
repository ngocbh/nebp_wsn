"""
File: sprim_mutation.py
Created by ngocjr7 on 2020-09-15 21:16
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

import numpy as np

class EPrimMutation(Mutation):

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
        
        ep_list = np.array(tree.get_energy_consumption_list())
        max_energy = np.amax(ep_list)
        most_used_nodes = np.where(ep_list == max_energy)
        slt_node = random_state.choice(most_used_nodes)

        tree.build_eprim_tree(max_energy, slt_node, random_state)
        ret_indv.encode(tree)

        return ret_indv
        
