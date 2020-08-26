"""
# Filename: networks.py
# Description:
# Created by ngocjr7 on [12-06-2020 15:55:08]
"""
from __future__ import absolute_import
from problems import SingleHopProblem, MultiHopProblem
from utils.point import distance
from utils.input import WusnInput, WusnConstants
from collections import deque
from geneticpython.models.tree import KruskalTree
import sys
import os
import math


class WusnNetwork(KruskalTree):
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
        super(WusnNetwork, self).__init__(number_of_vertices=self.node_count, root=0)

        self.initialize()

    def initialize(self):
        super(WusnNetwork, self).initialize()
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
                        parent[v] = u
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
                # else:
                #     self.edges.remove((i, 0))
                # self.adjacency[0].remove(i)
                # self.adjacency[i].remove(0)

        is_valid &= (max_depth - 1 <= self.max_hop)
        self.max_depth = max_depth
        is_valid &= all(visited[self.n+1:])
        self._is_valid &= is_valid

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
                    e = WusnConstants.k_bit * \
                        (WusnConstants.e_elec + WusnConstants.e_fs * d ** 2)
                else:
                    e = WusnConstants.k_bit * (self.num_childs[index] * (
                        WusnConstants.e_elec + WusnConstants.e_da) + WusnConstants.e_mp * d ** 4)

                max_energy_consumption = max(max_energy_consumption, e)

        return max_energy_consumption


class MultiHopNetwork(WusnNetwork):

    def calc_max_energy_consumption(self):

        def transmission_energy(k, d):
            d0 = math.sqrt(WusnConstants.e_fs / WusnConstants.e_mp)
            if d <= d0:
                return k * WusnConstants.e_elec + k * WusnConstants.e_fs * (d ** 2)
            else:
                return k * WusnConstants.e_elec + k * WusnConstants.e_mp * (d ** 4)

        max_energy_consumption = 0

        for index in range(1, self.node_count):
            if self.parent[index] != -1:
                d = distance(self._points[index],
                             self._points[self.parent[index]])
                e_t = transmission_energy(WusnConstants.k_bit, d)

                e_r = self.num_childs[index] * WusnConstants.k_bit * (WusnConstants.e_elec + WusnConstants.e_da) + (
                    index > self.n) * WusnConstants.k_bit * WusnConstants.e_da

                e = e_r + e_t

                max_energy_consumption = max(max_energy_consumption, e)

        return max_energy_consumption
