"""
File: parameters.py
Author: ngocjr7
Email: ngocjr7@gmail.com
Github: https://github.com/ngocjr7
Description: 
"""

import os
# "0.0.1" for run experimental arguments and "test" for testing
SH_MODEL = "test"
MODEL = "test"

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

class SHP(object):
    def __init__(self):
        self.SEED = 42
        self.POP_SIZE = 10
        self.SLT_SIZE = 10
        self.GENS = 2
        self.TOURNAMENT_SIZE = 5
        self.CRO_PROB = 0.9
        self.CRO_DI = 20
        self.MUT_PROB = 1.0/30
        self.MUT_DI = 5

    # @classmethod
    def load_model(self, model):
        if model == 'test':                 
            self.SEED = 42
            self.POP_SIZE = 10
            self.SLT_SIZE = 10
            self.GENS = 2
            self.TOURNAMENT_SIZE = 2
            self.CRO_PROB = 0.9
            self.CRO_DI = 5
            self.MUT_PROB = 1.0/100
            self.MUT_DI = 0.20
        elif model == '0.0.1':
            self.SEED = 42
            self.POP_SIZE = 300
            self.SLT_SIZE = 100
            self.GENS = 100
            self.TOURNAMENT_SIZE = 2
            self.CRO_PROB = 0.9
            self.CRO_DI = 5
            self.MUT_PROB = 1.0/100
            self.MUT_DI = 0.20
        elif model == '0.0.2':
            self.SEED = 42
            self.POP_SIZE = 400
            self.SLT_SIZE = 400
            self.GENS = 400
            self.TOURNAMENT_SIZE = 2
            self.CRO_PROB = 0.9
            self.CRO_DI = 5
            self.MUT_PROB = 1.0/100
            self.MUT_DI = 0.20
        elif model == '0.0.3':
            self.SEED = 42
            self.POP_SIZE = 100
            self.SLT_SIZE = 100
            self.GENS = 100
            self.TOURNAMENT_SIZE = 2
            self.CRO_PROB = 0.9
            self.CRO_DI = 5
            self.MUT_PROB = 1.0/30
            self.MUT_DI = 0.20
        elif model == '0.0.4':
            self.SEED = 42
            self.POP_SIZE = 400
            self.SLT_SIZE = 200
            self.GENS = 200
            self.TOURNAMENT_SIZE = 2
            self.CRO_PROB = 0.9
            self.CRO_DI = 20
            self.MUT_PROB = 1.0/100
            self.MUT_DI = 5

class MHP:
    def __init__(self):
        self.SEED = 42
        self.POP_SIZE = 10
        self.SLT_SIZE = 10
        self.GENS = 2
        self.TOURNAMENT_SIZE = 5
        self.X_PROB = 0.9
        self.CRO_DI = 20
        self.MUT_PROB = 1.0/30
        self.MUT_DI = 5
        self.ALPHA = 2
        self.BETA = 6

    def load_model(self, model):
        if model == 'test':                 
            self.SEED = 42
            self.POP_SIZE = 10
            self.SLT_SIZE = 10
            self.GENS = 2
            self.TOURNAMENT_SIZE = 2
            self.CRO_PROB = 0.9
            self.CRO_DI = 5
            self.MUT_PROB = 1.0/100
            self.MUT_DI = 0.20
            self.ALPHA = 2
            self.BETA = 6
        elif model == '0.0.1':
            self.SEED = 42
            self.POP_SIZE = 100
            self.SLT_SIZE = 100
            self.GENS = 100
            self.TOURNAMENT_SIZE = 2
            self.CRO_PROB = 0.9
            self.CRO_DI = 5
            self.MUT_PROB = 1.0/100
            self.MUT_DI = 0.20
            self.ALPHA = 2
            self.BETA = 6
        elif model == '0.0.2':
            self.SEED = 42
            self.POP_SIZE = 400
            self.SLT_SIZE = 400
            self.GENS = 400
            self.TOURNAMENT_SIZE = 2
            self.CRO_PROB = 0.9
            self.CRO_DI = 5
            self.MUT_PROB = 1.0/100
            self.MUT_DI = 0.20
            self.ALPHA = 2
            self.BETA = 6

# load_model(MODEL)
# load_model(MODEL)
