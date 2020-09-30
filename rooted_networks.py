
from __future__ import absolute_import
from problems import SingleHopProblem, MultiHopProblem
from utils.point import distance
from utils.input import WusnInput, WusnConstants as wc
from collections import deque
from geneticpython.models.tree import RootedTree
from geneticpython.utils import rset
import sys
import os
import math


class WusnNetwork(RootedTree):
    def __init__(self, problem: MultiHopProblem):
        self.m = problem._num_of_sensors
        self.n = problem._num_of_relays
        self.node_count = 1 + self.m + self.n
        self.potential_edges = problem._edges
        self.node_types = problem._node_types
        self.idx2edge = problem._idx2edge
        self.edge2idx = problem._edge2idx
        self._points = problem._points
        self.max_hop = problem.max_hop
        self.num_encoded_edges = problem._num_encoded_edges
        super(WusnNetwork, self).__init__(number_of_vertices=self.node_count, root=0, potential_edges=problem._idx2edge)

        self.initialize()

    def initialize(self):
        super(WusnNetwork, self).initialize()
        self.num_childs = [0 for i in range(self.node_count)]
        self.num_used_relays = 0
        self._is_valid = True

        for i in range(1, self.n+1):
            self.add_edge(0, i)

    def repair(self):
        visited = [False] * self.node_count
        is_valid = True
        parent = [-1] * self.node_count
        num_childs = [0] * self.node_count
        max_depth = 0

        def dfs(u: int, p: int, depth: int):
            ret = 1
            nonlocal max_depth
            visited[u] = True
            max_depth = max(max_depth, depth)
            
            for v in self.adjacency[u]:
                if v != p:
                    if visited[v]:
                        self._is_valid = False
                    else:
                        parent[v] = int(u)
                        ret += dfs(v, u, depth + 1)

            num_childs[u] = ret - 1
            return ret

        dfs(self.root, -1, 0)
        self.num_childs = num_childs
        self.num_used_relays = self.n
        for i in range(1, self.n + 1):
            if self.num_childs[i] == 0:
                self.num_used_relays -= 1
                self.parent[i] = -1
                self.num_childs[0] -= 1
                # if (0, i) in self.edges:
                #     self.edges.remove((0, i))
                # elif (i, 0) in self.edges:
                #     self.edges.remove((i, 0))
                # self.adjacency[0].remove(i)
                # self.adjacency[i].remove(0)

        is_valid &= (max_depth <= self.max_hop)
        self.max_depth = max_depth
        is_valid &= all(visited[self.n+1:])
        self._is_valid = is_valid

    @property
    def is_valid(self):
        return self._is_valid


class SingleHopNetwork(WusnNetwork):

    def calc_max_energy_consumption(self):
        max_energy_consumption = 0
        for index in range(1, self.node_count):
            if self.parent[index] != -1:
                d = distance(self._points[index],
                             self._points[self.parent[index]])
                if index > self.n:
                    e = wc.k_bit * \
                        (wc.e_elec + wc.e_fs * d ** 2)
                else:
                    e = wc.k_bit * (self.num_childs[index] * (
                        wc.e_elec + wc.e_da) + wc.e_mp * d ** 4)

                max_energy_consumption = max(max_energy_consumption, e)

        return max_energy_consumption

    def relay_energy(self, x, d):
        return wc.k_bit * (x * (wc.e_elec + wc.e_da) + wc.e_mp * d ** 4)

    def sensor_energy(self, d):
        return wc.k_bit * (wc.e_elec + wc.e_fs * d ** 2)

    def max_childs(self, max_energy, d):
        return int( (max_energy - wc.k_bit * wc.e_mp * d ** 4 ) / (wc.k_bit * wc.e_elec + wc.k_bit * wc.e_da) )


class MultiHopNetwork(WusnNetwork):

    def transmission_energy(self, k, d):
        d0 = math.sqrt(wc.e_fs / wc.e_mp)
        if d <= d0:
            return k * wc.e_elec + k * wc.e_fs * (d ** 2)
        else:
            return k * wc.e_elec + k * wc.e_mp * (d ** 4)

    def calc_max_energy_consumption(self):
        max_energy_consumption = 0

        for index in range(1, self.node_count):
            if self.parent[index] != -1:
                d = distance(self._points[index],
                             self._points[self.parent[index]])
                e_t = self.transmission_energy(wc.k_bit, d)

                e_r = self.num_childs[index] * wc.k_bit * (wc.e_elec + wc.e_da) + (
                    index > self.n) * wc.k_bit * wc.e_da

                e = e_r + e_t

                max_energy_consumption = max(max_energy_consumption, e)

        return max_energy_consumption 

    def max_childs(self, i, max_energy, d):
        y = (i > self.n) 
        return int( (max_energy - self.transmission_energy(wc.k_bit, d) - y * wc.k_bit * wc.e_da) \
                   / (wc.k_bit * wc.e_elec + wc.k_bit * wc.e_da) )

    def add_child(self, u, max_childs):
        max_childs[u] -= 1
        if u != self.root:
            self.add_child(self.parent[u], max_childs)

    def update_max_childs(self, u, max_childs):
        if u == self.root:
            return 1000000
        x = self.update_max_childs(self.parent[u], max_childs)
        max_childs[u] = min(x, max_childs[u])
        return max_childs[u]

    def run_algorithm(self, C, A, adj, max_childs, max_energy, random_state, used_max_childs=True):
        while len(C) < self.node_count and len(A) > 0:
            u, v = A.random_choice(random_state)
            A.remove((u, v))
            if v in C:
                continue
            if used_max_childs:
                self.update_max_childs(u, max_childs)
            if max_childs[u] > 0 or not used_max_childs:
                self.add_edge(u, v)
                C.add(v)
                if used_max_childs:
                    self.add_child(u, max_childs)
                    d = distance(self._points[u], self._points[v])
                    max_childs[v] = min(max_childs[u], self.max_childs(v, max_energy, d))
                for w in adj[v]:
                    if w not in C:
                        A.add((v, w))
    
    def build_cprim_tree(self, max_energy, uedges, random_state=None):
        max_childs = [0] * self.node_count
        self.initialize()
        C = set()
        A = rset()
        C.add(0)

        # phase 1: init tree with union edges of 2 parents, with max_energy upper_bound
        # print('phase 1')
        uadj = self.get_adjacency(uedges)
        for u in uadj[0]:
            C.add(u)
            d = distance(self._points[0], self._points[u])
            max_childs[u] = self.max_childs(u, max_energy, d)
            if max_childs[u] > 0:
                for v in uadj[u]:
                    A.add((u, v))

        self.run_algorithm(C, A, uadj, max_childs, max_energy, random_state)

        # print(len(self.edges))
        # phase 2: continue building tree with full edges
        if len(C) < self.node_count:
            # print('phase 2')
            for u in C:
                if max_childs[u] > 0:
                    for v in self.potential_adj[u]:
                        if v not in C:
                            A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state)

        # print(len(self.edges))
        # phase 3: continue building tree without upper_bound

        if len(C) < self.node_count:
            # print('phase 3')
            for u in C:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, used_max_childs=False)
        # print(len(self.edges))
        self.repair()

    def build_mprim_tree(self, max_energy, used_relays, edges, random_state):
        max_childs = [0] * self.node_count
        self.initialize()
        C = set()
        A = rset()
        C.add(0)

        # nr = 0
        # for e in used_relays:
        #     if e:
        #         nr += 1;
        # print('num used relays = ', nr)
        # phase 1: use edges first, keep locality
        adj = self.get_adjacency(edges)
        for u in range(1, self.n+1):
            C.add(u)
            if used_relays[u]:
                d = distance(self._points[0], self._points[u])
                max_childs[u] = self.max_childs(u, max_energy, d)
                if max_childs[u] > 0:
                    for v in adj[u]:
                        A.add((u, v))

        self.run_algorithm(C, A, adj, max_childs, max_energy, random_state)
        # print(len(self.edges))

        for u in C:
            if max_childs[u] > 0:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state)
        # print(len(self.edges))

        # # phase 2: continue building tree with full edges
        if len(C) < self.node_count:
            # print('phase 2')
            for u in C:
                if u > self.n or used_relays[u]:
                    for v in self.potential_adj[u]:
                        if v not in C:
                            A.add((u, v))
        # print(len(self.edges))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, used_max_childs=False)
        
        # phase 3: continue building tree without upper_bound

        if len(C) < self.node_count:
            for u in C:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, used_max_childs=False)
        # print(len(self.edges))
        self.repair()
        # print(self.num_used_relays)




        
        
