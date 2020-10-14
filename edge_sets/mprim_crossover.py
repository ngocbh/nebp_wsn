"""
File: mprim_crossover.py
Created by ngocjr7 on 2020-09-12 15:41
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

from geneticpython.core.operators import Crossover
from geneticpython.core.individual import Individual
from geneticpython.utils.validation import check_random_state
from geneticpython.models.tree import Tree


class MPrimCrossover(Crossover):
    __EPS = 1e-8
    no_improved = 0

    def cross(self, father: Individual, mother: Individual, random_state=None):
        random_state = check_random_state(random_state)
        do_cross = True if random_state.random() <= self.pc else False

        children = father.clone(), mother.clone()
        if not do_cross:
            return children

        trees = children[0].decode(), children[1].decode()
        if not (isinstance(trees[0], Tree) and isinstance(trees[1], Tree)):
            raise ValueError(f"The PrimCrossover is only used on the individual that \
                             decodes to an instance of Tree. \
                             got father type: {type(trees[0])} and mother type {type(trees[1])}")

        edge_union = set() 
        potential_adj = [list() for _ in range(trees[0].number_of_vertices)]
        for i in range(2):
            for u, v in trees[i].edges:
                if (v, u) not in edge_union:
                    edge_union.add((u, v))
                    potential_adj[u].append(v)
                    potential_adj[v].append(u)

        energy0 = trees[0].calc_max_energy_consumption()
        trees[0].build_cprim_tree(energy0, edge_union, random_state)
        if trees[0].calc_max_energy_consumption() - energy0 < 0:
            MPrimCrossover.no_improved += 1

        energy1 = trees[1].calc_max_energy_consumption()
        trees[1].build_cprim_tree(energy1, edge_union, random_state)
        if trees[1].calc_max_energy_consumption() - energy1 < 0:
            MPrimCrossover.no_improved += 1

        # print(trees[0].edges)
        children[0].encode(trees[0])
        children[1].encode(trees[1])

        return children[0], children[1]
