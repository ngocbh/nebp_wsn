
from typing import List, Union, Callable, Dict
from random import Random
from tqdm.auto import tqdm
import copy

from geneticpython.core import Individual, Population, Pareto
from geneticpython.core.operators import Selection, Crossover, Mutation
from geneticpython.core.operators import Replacement, RankReplacement, TournamentSelection
from geneticpython.engines.geneticengine import GeneticEngine
from geneticpython.callbacks import CallbackList, Callback, History
from geneticpython.engines.multi_objective.multi_objective_engine import MultiObjectiveEngine, is_dominated
from geneticpython.utils.validation import check_random_state
from geneticpython.engines import MultiObjectiveEngine, NSGAIIEngine

class HMOEAEngine(MultiObjectiveEngine):
    def __init__(self, min_relays, max_relays, max_step, **kwargs):
        self.max_step = max_step
        self.step = max(int((max_relays - min_relays + 1) / max_step), 1)
        self.max_step = int((max_relays - min_relays + 1) / self.step)
        self.min_relays = min_relays
        self.max_relays = max_relays
        super(HMOEAEngine, self).__init__(**kwargs)

    def do_initialization(self) -> List[Individual]:
        # Injecting nondominated_rank and crowding_distance for template individuals
        # This just makes it easier to understand
        self.population.individual_temp.nondominated_rank = None
        self.population.individual_temp.crowding_distance = None
        # init population
        population = self.population.init_population(self.random_state)

        return population

    def update_pareto(self, pareto, indv):
        objs = indv.compute_objectives(self.min_relays, self.max_relays, self.step)
        for obj1, obj2 in objs:
            if obj1 != float('inf'):
                if obj2 < pareto[int(obj1)][0]:
                    pareto[int(obj1)] = (obj2, indv)


    def do_evaluation(self, population: List[Individual]) -> List[Individual]:
        population = NSGAIIEngine.sort(population, self.random_state)
        return population

    def do_reproduction(self, mating_population: List[Individual]) -> List[Individual]:
        childs = []
        for i in range(0, len(mating_population), 2):
            childs_temp = self.crossover.cross(father=mating_population[i],
                                               mother=mating_population[i+1],
                                               random_state=self.random_state)
            childs.extend(childs_temp)

        for i in range(len(childs)):
            childs[i] = self.mutation.mutate(childs[i], random_state=self.random_state)

        return childs

    def cfs_operator(self, p1, p2):
        ret = 0
        for i in range(len(p1.objs)):
            if p1.objs[i][1] < p2.objs[i][1]:
                ret += 1
            elif p1.objs[i][1] > p2.objs[i][1]:
                ret -= 1
        return ret

    def cfs_comparator(self, p1, p2):
        cfs = self.cfs_operator(p1, p2)
        if cfs > 0:
            return -1
        elif cfs < 0:
            return 1
        else:
            return 0


    def do_selection(self) -> List[Individual]:
        try:
            mating_population = self.selection.select(self.selection_size,
                                                      population=self.population.individuals,
                                                      comparator=self.cfs_comparator,
                                                      random_state=self.random_state)
        except TypeError as err:
            raise TypeError(" {}\nSelection does not support nondominated comparator,\n \
                    TournamentSelection is recommended".format(err))

        return mating_population

    def do_replacement(self, offspring_population) -> List[Individual]:
        for ofs in offspring_population:
            max_cfs = len(ofs.objs)
            for i, indv in enumerate(self.population.individuals):
                cfs = self.cfs_operator(ofs, indv)
                if cfs == max_cfs or (cfs > 0 and self.random_state.random() < 0.95):
                    self.population.individuals[i] = ofs

    def get_sim_solutions(self):
        sols = []
        for obj1, (obj2, indv) in enumerate(self.pareto):
            if obj2 != float('inf'):
                sols.append((obj1, obj2))
        return sols


    def get_pareto_front(self) -> Pareto:
        pareto = []
        min_obj2 = float('inf')
        for obj1, (obj2, indv) in enumerate(self.pareto):
            if obj2 < min_obj2:
                indv1 = copy.deepcopy(indv)
                indv1.main_obj = (obj1, obj2)
                pareto.append(indv1)
                min_obj2 = obj2

        return pareto

    def get_sim_pareto(self):
        sim_par = []
        pareto = self.get_pareto_front()
        for indv in pareto:
            sim_par.append(indv.main_obj)
        return sim_par

    def _update_logs(self, logs):
        logs = logs or {}
        pareto_front = self.get_sim_pareto()
        logs.update({'pareto_front': pareto_front})
        solution = self.get_sim_solutions()
        logs.update({'solutions': solution})
        return logs

    def run(self, generations: int = None) -> History:
        if generations is not None:
            self.generations = generations

        logs = None
        self.history = History()
        self.selection_size = int(10000 / self.generations / self.max_step)
        self.selection_size += self.selection_size % 2

        self.pareto = [(float('inf'), None)] * (self.max_relays + 1)

        print('Initializing...', flush=True)
        self.population.individuals = self.do_initialization()

        for indv in self.population.individuals:
            self.update_pareto(self.pareto, indv)
        print('Finished Initialization!')

        self._update_metrics()
        logs = self._update_logs(logs)
        self.history.history.append(copy.deepcopy(logs))

        self.progbar = tqdm(range(self.generations))

        for gen in range(self.generations):

            mating_population = self.do_selection()
            offspring_population = self.do_reproduction(mating_population)
            
            for indv in offspring_population:
                self.update_pareto(self.pareto, indv)

            self.do_replacement(offspring_population)

            self._update_metrics()
            logs = self._update_logs(logs)
            self.history.history.append(copy.deepcopy(logs))

            self.progbar.update()

        self.progbar.close()
        print('Done!')

        return self.history
