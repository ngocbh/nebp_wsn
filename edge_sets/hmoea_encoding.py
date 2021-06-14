
from geneticpython.models import PermutationIndividual
from geneticpython.models.tree import Tree
from geneticpython.utils.validation import check_random_state
from geneticpython.utils.rset import rset
from typing import List, Tuple
import numpy as np
from copy import deepcopy
from queue import PriorityQueue, Queue


class HMOEAEncoder(PermutationIndividual):
    def __init__(self, max_hop, num_sensors, num_relays, adj, chromosome=None, solution=None):
        super(HMOEAEncoder, self).__init__(num_sensors, start=num_relays+1)
        self.max_hop = max_hop
        self.num_sensors = num_sensors
        self.num_relays = num_relays
        self.num_vertices = num_sensors + num_relays + 1
        self.adj = adj
        self.solution = solution
        self.nondominated_rank = None
        self.crowding_distance = None
        self.objs = None
        self.main_obj = None
        if chromosome is not None:
            self.chromosome = chromosome

    def compute_objectives(self, min_relays, max_relays, step):
        self.objs = []
        for rl in range(min_relays, max_relays + 1, step):
            self.decode(rl + step)
            obj1 = self.solution.get_number_of_used_relays()
            obj2 = self.solution.calc_max_energy_consumption()
            self.objs.append((obj1, obj2))
        return self.objs


    def clone(self):
        num_sensors = self.num_sensors
        num_relays = self.num_relays
        max_hop = self.max_hop
        chromosome = deepcopy(self.chromosome)
        solution = self.solution.clone()

        return HMOEAEncoder(max_hop, num_sensors, num_relays, self.adj, chromosome, solution)

    def decode(self, max_relays=None):
        seed = np.sum(self.chromosome.genes * np.arange(1, self.chromosome.length*3, 3)).item()
        random_state = check_random_state(seed)
        max_relays = max_relays or self.num_relays
        max_hop = self.max_hop or 12

        self.solution.initialize()
        _is_valid = True
        
        num_used_relays = 0
        relays_mark = np.zeros(self.num_relays)
        order = np.zeros(self.num_vertices)
        # print(self.chromosome.genes)
        order[self.chromosome.genes] = np.arange(1, self.num_sensors + 1)
        # print(order)
        num_children = np.zeros(self.num_vertices)

        # Set of connected nodes
        C = set()
        # eligible edges
        A = PriorityQueue()

        # Init tree
        C.add(0)
        for v in range(1, self.num_relays + 1):
            num_children[0] += 1
            C.add(v)

        for u in C:
            for d, v in self.adj[u]:
                A.put((order[v], d * random_state.random(), v, u))

        # print(A.queue)

        _is_valid = True

        while not A.empty():
            orv, duv, v, u = A.get()

            if v not in C:
                if 0 < u <= self.num_relays and num_children[u] == 0 and num_used_relays >= max_relays:
                    continue
                if self.solution.try_add_edge(u, v) and self.solution.depth[u] + 1 <= max_hop :
                    _is_valid &= self.solution.add_edge(u, v)
                    C.add(v)
                    num_used_relays += (num_children[u] == 0 and 0 < u <= self.num_relays)
                    num_children[u] += 1
                    for dvw, w in self.adj[v]:
                        if w not in C:
                            A.put((order[w], dvw * random_state.random(), w, v))

        notC = set()
        for u in range(self.num_vertices):
            if u not in C:
                notC.add(u)

        for u in C:
            for duv, v in self.adj[u]:
                if v in notC:
                    A.put((order[v], duv * random_state.random(), v, u))
        
        # print(notC)

        # print(self.solution.edges)
        # print(len(self.solution.edges))

        while not A.empty():
            orv, duv, v, u = A.get()

            if v not in C:
                self.solution.add_edge(u, v)
                C.add(v)
                num_used_relays += (num_children[u] == 0 and 0 < u <= self.num_relays)
                num_children[u] += 1
                for dvw, w in self.adj[v]:
                    if w not in C:
                        A.put((order[w], dvw * random_state.random(), w, v))

        # print(self.solution.edges)
        # print(len(self.solution.edges))

        self.solution._is_valid = _is_valid
        self.solution.repair()

        return self.solution

    def encode(self, solution: Tree, random_state=None):
        order = [i for i in range(solution.number_of_vertices-1)]

        adj = solution.get_adjacency()
        qu = Queue()
        for i in range(1, self.num_relays+1):
            qu.put(i)

        visited = np.zeros(self.num_vertices)
        visited[0] = 1
        genes = []
        while not qu.empty():
            u = qu.get()
            visited[u] = 1

            if u > self.num_relays:
                genes.append(u)

            for v in adj[u]:
                if visited[v] != 1:
                    qu.put(v)
            

        self.update_genes(genes)
        self.solution = solution

