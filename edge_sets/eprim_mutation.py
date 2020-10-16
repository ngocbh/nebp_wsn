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
    __EPS = 1e-8
    no_improved = 0

    def __init__(self, pm, max_hop=None):
        self.max_hop = max_hop
        super(EPrimMutation, self).__init__(pm=pm)

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
        most_used_nodes = np.where(ep_list == max_energy)[0]
        slt_node = random_state.choice(most_used_nodes)
        print("Running mutation:")
        print("num_used_relays, max_energy, most_used_nodes, slt_node = ", tree.num_used_relays, max_energy, most_used_nodes, slt_node)

        tree.build_eprim_tree(max_energy, slt_node, random_state, max_hop=self.max_hop)

        if tree.calc_max_energy_consumption() - max_energy < - EPrimMutation.__EPS:
            print("Good move")
            EPrimMutation.no_improved += 1
        print("New used relays, New energy = ", tree. tree.calc_max_energy_consumption())

        ret_indv.encode(tree)

        return ret_indv
        
