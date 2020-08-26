"""
File: kruskal_mutation.py
Created by ngocjr7 on 2020-08-21 18:42
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

from __future__ import absolute_import

from geneticpython.core.individual import Individual
from geneticpython.core.operators import Mutation

class KruskalMutation(Mutation):
    def __init__(self, pm : float):
        super(KruskalMutation, self).__init__(pm=pm)

    def mutate(self, indv : Individual, random_state=None) -> Individual:
        pass
