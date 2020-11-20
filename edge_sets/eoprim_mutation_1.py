
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

class EOPrimMutation1(Mutation):
    __EPS = 1e-8
    no_improved = 0

    def __init__(self, pm, max_hop=None, backup_mutation=lambda indv, random_state: None):
        self.max_hop = max_hop
        self.backup_mutation = backup_mutation
        super(EOPrimMutation1, self).__init__(pm=pm)

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
        
        # print("begin" + "="*10)
        ep_list = np.array(tree.get_energy_consumption_list())
        args = np.argsort(ep_list)
        max_energy = tree.calc_max_energy_consumption()
        if max_energy == float('inf'):
            tree.build_depth_constraint_prim_tree(random_state, self.max_hop)
            ret_indv.encode(tree)
            # print("===> Conclusion: Invalid tree, random another one")
        else:
            most_used_nodes, = np.where( np.abs(ep_list - max_energy) < 1e-10)
            slt_node = random_state.choice(most_used_nodes)
            # print("Running mutation:")
            # print("num_used_relays, max_energy, most_used_nodes, slt_node = ", tree.num_used_relays, max_energy, most_used_nodes, slt_node)

            improved = tree.build_energy_oriented_prim_tree_1(max_energy, slt_node, random_state, self.max_hop)

            if improved:
                EOPrimMutation1.no_improved += 1
                ret_indv.encode(tree)
                # print("===> Conclusion: Improved {} {} -> {}".format(tree.get_number_of_used_relays(), max_energy, tree.calc_max_energy_consumption()))
            # else:
                # ret_indv = self.backup_mutation.mutate(indv, random_state)
                # print("===> Conclusion: Not improved, use backup_mutation")

        # print("end: number of improved = {}".format(EOPrimMutation.no_improved) + "="*10)
        return ret_indv
        
