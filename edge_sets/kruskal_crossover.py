"""
File: kruskal_crossover.py
Created by ngocjr7 on 2020-08-16 21:32
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""
from __future__ import absolute_import

from typing import Tuple

from geneticpython.core.operators import Crossover
from geneticpython.core.individual import Individual

class KruskalCrossover(Crossover):
    def __init__(self, pc : float):
        super(KruskalCrossover, self).__init__(pc=pc)

    def cross(self, father : Individual, mother : Individual, random_state=None) -> Tuple[Individual]:
        father_network = father.decode()
        mother_network = mother.decode()

        
