
from geneticpython.core.individual import Individual
from geneticpython.core.operators.mutation import Mutation
from geneticpython.utils.validation import check_random_state

from typing import List
from itertools import accumulate
from bisect import bisect_right

class MyMutationCompact(Mutation):
    def __init__(self, mutation_list : List[Mutation] = None, t_list : List[int] = None):
        mutation_list = mutation_list or []
        t_list = t_list or []

        if len(mutation_list) != len(t_list):
            raise ValueError(f'mutation_list and pm_list must have same length,\n\
                             given mutation_list:{len(mutation_list)} and pm_list:{len(t_list)}')

        if len(t_list) > 0:
            self.acc_t = list(accumulate(t_list))
        else:
            self.acc_t = []

        self.mutation_list = mutation_list
        self.no_mutating = 0
        self.idx = 0

    def add_mutation(self, mutation : Mutation, t : int):
        self.mutation_list.append(mutation)
        s = self.acc_t[-1] if len(self.acc_t) > 0 else 0
        self.acc_t.append(s + t)

    def mutate(self, indv : Individual, random_state=None):
        random_state = check_random_state(random_state)

        if self.no_mutating > self.acc_t[self.idx] and self.idx < len(self.acc_t) - 1:
            self.idx += 1
        self.no_mutating += 1

        ret_indv = self.mutation_list[self.idx].mutate(indv, random_state)
        return ret_indv
