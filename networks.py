"""
# Filename: networks.py
# Description:
# Created by ngocjr7 on [12-06-2020 15:55:08]
"""
from __future__ import absolute_import
from problems import SingleHopProblem, MultiHopProblem
from utils.point import distance
from utils.input import WusnInput, WusnConstants as wc
from collections import deque
from geneticpython.models.tree import KruskalTree
import sys
import os
import math


class WusnKruskalNetwork(KruskalTree):

    def __init__(self, problem: MultiHopProblem):
        self.problem = problem
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
        # super(WusnKruskalNetwork, self).__init__(number_of_vertices=self.node_count,
        #                                          root=0, potential_edges=problem._idx2edge, init_method='KruskalRST')

        self.number_of_vertices = self.node_count
        self.root = 0

        self.potential_edges = problem._idx2edge
        self.potential_adj = problem.potential_adj
        self.set_initialization_method('KruskalRST')
        self.initialize()

    def initialize(self):
        super(WusnKruskalNetwork, self).initialize()
        self.parent = [-1 for i in range(self.node_count)]
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
        self.parent = parent
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
        is_valid &= (len(self.edges) == self.number_of_vertices-1)

        potential_edge_set = set(self.potential_edges)
        for u, v in self.edges:
            if (u == 0 and v <= self.n) or (v == 0 and u <= self.n):
                continue
            if (u, v) not in potential_edge_set \
                    and (v, u) not in potential_edge_set:
                is_valid &= False

        edge_set = set(self.edges)
        for i in range(1, self.n+1):
            if is_valid and (0, i) not in edge_set and (i, 0) not in edge_set:
                raise ValueError('Network is not valid but is_valid is true')

        self._is_valid = is_valid
        if self.num_childs[0] - self.m != self.num_used_relays:
            print(self.edges)
            print(self.num_childs)
            print(self.parent)
            print(self.num_used_relays)
            raise ValueError('oasfdjo')

    @property
    def is_valid(self):
        return self._is_valid


class SingleHopNetwork(WusnKruskalNetwork):

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


class MultiHopNetwork(WusnKruskalNetwork):

    def clone(self):
        ret = MultiHopNetwork(self.problem)
        return ret

    def transmission_energy(self, k, d):
        d0 = math.sqrt(wc.e_fs / wc.e_mp)
        if d <= d0:
            return k * wc.e_elec + k * wc.e_fs * (d ** 2)
        else:
            return k * wc.e_elec + k * wc.e_mp * (d ** 4)

    def energy_consumption(self, x, y, d):
        e_t = self.transmission_energy(wc.k_bit, d)
        # e_r = x * wc.k_bit * (wc.e_elec + wc.e_da) + y * wc.k_bit * wc.e_da
        e_r = wc.k_bit * wc.e_elec
        e = x * e_r + (x + y) * e_t
        return e

    def get_energy_consumption_list(self):
        ret = [0]
        for index in range(1, self.node_count):
            if self.parent[index] != -1:
                d = distance(self._points[index],
                             self._points[self.parent[index]])
                e = self.energy_consumption(self.num_childs[index], (index > self.n), d)
                ret.append(e)
            else:
                ret.append(0)

        return ret

    def get_number_of_used_relays(self):
        if self.is_valid:
            return self.num_used_relays
        else:
            return float('inf')

    def calc_max_energy_consumption(self):
        ep_list = self.get_energy_consumption_list()
        if self.is_valid:
            return max(ep_list) 
        else:
            return float('inf')
