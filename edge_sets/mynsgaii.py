from __future__ import absolute_import

from geneticpython.engines import NSGAIIEngine

class MyNSGAIIEngine(NSGAIIEngine):
    def do_reproduction(self, mating_population):
        childs = []
        for i in range(0, len(mating_population), 2):
            childs_temp = self.crossover.cross(father=mating_population[i],
                                               mother=mating_population[i+1],
                                               random_state=self.random_state)
            childs.extend(childs_temp)

        for i in range(len(mating_population)):
            child_temp = self.mutation.mutate(mating_population[i], random_state=self.random_state)
            childs.append(child_temp)

        return childs
