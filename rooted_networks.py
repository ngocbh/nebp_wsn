
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
import copy


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
        is_valid &= (len(self.edges) == self.number_of_vertices-1)

        for i in range(1, self.n+1):
            if is_valid and (0, i) not in self.edges and (i, 0) not in self.edges:
                raise ValueError('Network is not valid but is_valid is true')

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

    def energy_consumption(self, x, y, d):
        e_t = self.transmission_energy(wc.k_bit, d)
        e_r = x * wc.k_bit * (wc.e_elec + wc.e_da) + y * wc.k_bit * wc.e_da
        e = e_r + e_t
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

    def max_childs(self, i, max_energy, d, strict_lower=True):
        if max_energy > 10000000:
            return 10000000

        y = (i > self.n) 
        nmc = (max_energy - self.transmission_energy(wc.k_bit, d) - y * wc.k_bit * wc.e_da) \
                   / (wc.k_bit * wc.e_elec + wc.k_bit * wc.e_da)
        if abs(nmc - int(nmc)) < 1e-10 and strict_lower:
            return int(nmc) - 1
        else:
            return int(nmc)

    def add_child(self, u, max_childs):
        max_childs[u] -= 1
        if u != self.root:
            self.add_child(self.parent[u], max_childs)

    def update_max_childs(self, u, max_childs, _print=False):
        if u == self.root:
            return 100000000
        x = self.update_max_childs(self.parent[u], max_childs, _print)
        # if _print:
            # print("update_max_childs: {}: {} {}".format(u, max_childs[u], x))
        max_childs[u] = min(x, max_childs[u])
        return max_childs[u]

    def run_algorithm(self, C, A, adj, max_childs, max_energy, random_state, used_max_childs=True, strict_lower=True, max_hop=None):
        while len(C) < self.node_count and len(A) > 0:
            # print(max_childs[254])
            u, v = A.random_choice(random_state)
            A.remove((u, v))
            if v in C or (max_hop is not None and self.depth[u] >= max_hop):
                # if v not in C:
                #     print("max_hop bound {}: {}".format(u, self.depth[u]))
                continue

            if used_max_childs:
                self.update_max_childs(u, max_childs, False)
            if not used_max_childs or max_childs[u] > 0:
                self.add_edge(u, v)
                # print("add edge {} {}".format(u, v))
                C.add(v)
                if used_max_childs:
                    self.add_child(u, max_childs)
                    d = distance(self._points[u], self._points[v])
                    max_childs[v] = min(max_childs[u], self.max_childs(v, max_energy, d, strict_lower))
                for w in adj[v]:
                    if w not in C:
                        A.add((v, w))
            # else:
            #     if used_max_childs:
            #         print("max_child bound {}-{}: {}".format(u, v, max_childs[u]))

    def build_depth_constraint_prim_tree(self, random_state, max_hop=None):
        self.initialize()
        # Set of connected nodes
        C = set()
        # eligible edges
        A = rset()

        for u in range(self.number_of_vertices):
            if self.parent[u] != -1:
                C.add(u)
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, [], float('inf'), random_state, used_max_childs=False, max_hop=max_hop)

        if len(C) < self.number_of_vertices:
            # print('phase 3')
            for u in C:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, [], float('inf'), random_state, used_max_childs=False, max_hop=None)

        self.repair()
    
    def build_xprim_tree(self, max_energy, uedges, random_state, max_hop=None):
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
            max_childs[u] = self.max_childs(u, max_energy, d, strict_lower=True)
            if max_childs[u] > 0:
                for v in uadj[u]:
                    A.add((u, v))

        self.run_algorithm(C, A, uadj, max_childs, max_energy, random_state, strict_lower=True, max_hop=max_hop)

        # print(len(self.edges))
        # phase 2: continue building tree with full edges
        if len(C) < self.node_count:
            # print('phase 2')
            for u in C:
                if max_childs[u] > 0:
                    for v in self.potential_adj[u]:
                        if v not in C:
                            A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, strict_lower=True, max_hop=max_hop)

        # print(len(self.edges))
        # phase 3: continue building tree without upper_bound

        if len(C) < self.node_count:
            # print('phase 3')
            for u in C:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, used_max_childs=False, strict_lower=True, max_hop=None)
        # print(len(self.edges))
        self.repair()


    def build_relay_oriented_prim_tree(self, max_energy, used_relays, edges, random_state, max_hop=None):
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
                max_childs[u] = self.max_childs(u, max_energy, d, strict_lower=False)
                if max_childs[u] > 0:
                    for v in adj[u]:
                        A.add((u, v))

        self.run_algorithm(C, A, adj, max_childs, max_energy, random_state, strict_lower=False, max_hop=max_hop)
        # print(len(self.edges))

        for u in C:
            if max_childs[u] > 0:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, strict_lower=False, max_hop=max_hop)
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

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, used_max_childs=False, strict_lower=False, max_hop=max_hop)
        # phase 3: continue building tree without upper_bound

        if len(C) < self.node_count:
            for u in C:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, used_max_childs=False, strict_lower=False, max_hop=None)
        # print(len(self.edges))
        self.repair()
        # print(self.num_used_relays)

    def decompose_tree(self, edges, deleted_node, max_energy, strict_lower=True):
        visited = [False] * self.number_of_vertices
        used_edges = set()
        max_childs = [-float('inf')] * self.number_of_vertices
        max_childs[0] = 1000000

        def dfs(u, deleted_node, adj, max_energy, max_childs, strict_lower):
            visited[u] = True

            for v in adj[u]:
                if v != deleted_node and not visited[v]:
                    if (v, u) not in used_edges:
                        used_edges.add((u, v))
                    d = distance(self._points[u], self._points[v])
                    max_childs[v] = self.max_childs(v, max_energy, d, strict_lower=True)
                    self.add_child(u, max_childs)

                    dfs(v, deleted_node, adj, max_energy, max_childs, strict_lower)

        adj = self.get_adjacency(edges)
        dfs(0, deleted_node, adj, max_energy, max_childs, strict_lower)
        depths = [0] * self.number_of_vertices
        for i in range(self.number_of_vertices):
            if max_childs[i] != -float('inf'):
                depths[i] = self.depth[i]

        return list(used_edges), max_childs, depths

    def build_energy_oriented_prim_tree(self, max_energy, slt_node, random_state, max_hop=None):
        edges = copy.deepcopy(self.edges)

        used_edges, max_childs, depths = self.decompose_tree(edges, slt_node, max_energy, True)
        # print(used_edges)
        # print(max_childs)
        # print(depths)
        # print(max_hop)
        free_vertices = set()
        for i in range(self.number_of_vertices):
            if max_childs[i] == -float('inf'):
                free_vertices.add(i)
            else:
                self.update_max_childs(i, max_childs)

        used_relays = [False] * (self.n + 1) 
        for i in range(1, self.n+1):
            used_relays[i] = (self.num_childs[i] != 0)
        
        sub_edges = list()
        for u, v in self.potential_edges:
            if u in free_vertices and v in free_vertices:
                sub_edges.append((u, v))


        sub_adj = self.get_adjacency(sub_edges)
        # print(sub_adj)
        t = 0
        low = [0] * self.number_of_vertices
        num = [0] * self.number_of_vertices
        is_articulation = [False] * self.number_of_vertices

        def tarjan(u, p):
            nonlocal t, low, num, is_articulation, sub_adj
            t += 1
            low[u], num[u] = t, t
            num_child = 0

            for v in sub_adj[u]:
                if v != p:
                    if num[v] != 0:
                        low[u] = min(low[u], num[v])
                    else:
                        tarjan(v, u)
                        num_child += 1
                        low[u] = min(low[u], low[v])
                        
                        if u == p:
                            if num_child >= 2:
                                is_articulation[u] = True
                        else:
                            if low[v] >= num[u]:
                                is_articulation[u] = True


        tarjan(slt_node, slt_node)
        if slt_node <= self.n:
            is_articulation[slt_node] = True

        non_articulation_points = set()
        for v in free_vertices:
            if not is_articulation[v]:
                non_articulation_points.add(v)

        # print(non_articulation_points)
        open_vertices = set()
        for i in range(1, self.number_of_vertices):
            if i not in free_vertices and max_childs[i] > 0 and depths[i] < max_hop:
                open_vertices.add(i)

        # print(open_vertices)

        potential_added_edges = []
        for u, v in self.potential_edges:
            if u in non_articulation_points and v in open_vertices:
                potential_added_edges.append((v, u))
            elif u in open_vertices and v in non_articulation_points:
                potential_added_edges.append((u, v))
        
        # print(potential_added_edges)
        if len(potential_added_edges) == 0:
            # print("len(potential_added_edges) == 0")
            # self.build_eprim_tree(max_energy, slt_node, random_state, max_hop)
            return False

        idx = random_state.randint(0, len(potential_added_edges))
        slt_edge = potential_added_edges[idx]
        # print(slt_edge)

        parent_slt_node = self.parent[slt_node]
        self.initialize()

        sorted_used_edges = self.sort_by_bfs_order(used_edges)
        for u, v in sorted_used_edges:
            self.add_edge(u, v)

        # print(used_edges)
        used_edges.append((parent_slt_node, slt_node))
        d = distance(self._points[slt_node], self._points[parent_slt_node])
        max_childs[slt_node] = self.max_childs(slt_node, max_energy, d, strict_lower=True) 
        # print(slt_node, max_childs[slt_node], max_energy)

        self.add_child(parent_slt_node, max_childs)
        self.add_edge(parent_slt_node, slt_node)

        used_edges.append(slt_edge)
        d = distance(self._points[slt_edge[0]], self._points[self.parent[slt_edge[1]]])
        max_childs[slt_edge[0]] = self.max_childs(slt_edge[0], max_energy, d, strict_lower=True) 
        self.add_child(slt_edge[0], max_childs)
        self.add_edge(slt_edge[0], slt_edge[1])
        
        new_sub_edges = []
        for u, v in sub_edges:
            if u != slt_edge[1] and v != slt_edge[1]:
                new_sub_edges.append((u, v))
        new_sub_adj = self.get_adjacency(new_sub_edges)
        # print(new_sub_adj)
        C = set()
        A = rset()
        # C.add(slt_node)
        # for v in new_sub_adj[slt_node]:
        #     A.add((slt_node, v))

        # print(max_childs[238])
        # print(max_childs[87])
        for i in range(self.number_of_vertices):
            if max_childs[i] != -float('inf'):
                C.add(i)
            elif i <= self.n and max_childs[i] == -float('inf'):
                C.add(i)
                d = distance(self._points[0], self._points[i])
                max_childs[i] = self.max_childs(i, max_energy, d, strict_lower=True)
            else:
                max_childs[i] = 0

        C.add(slt_edge[1])
        # print(free_vertices)
        for u in C:
            if max_childs[u] > 0 and u == slt_node:
                # print(u)
                for v in self.potential_adj[u]:
                    if v not in C:
                        # print(u, v)
                        A.add((u, v))

        # print(max_childs)
        # print(max_childs[87])
        x = set()
        for i in range(self.number_of_vertices):
            if i not in C:
                x.add(i)
        # print(x, len(x))
        # print(269 in C)
        # print(C, A)

        # print(max_childs[254])
        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, strict_lower=True, max_hop=max_hop)

        # print(free_vertices.difference(C))
        # print(""len(self.edges))
        # if len(self.edges) == self.number_of_vertices - 1:
        #     print(" -> Built get better energy tree. Selected node: {} | Selected edge: {}".format(slt_node, slt_edge))
        # else:
        #     print(" -> Bad tree, rebuild. Selected node: {} | Selected edge: {}".format(slt_node, slt_edge))

        if len(C) < self.node_count:
            for u in C:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, used_max_childs=False, strict_lower=True, max_hop=None)
        
        self.repair()

        # print(C, A)

        return True
        
    # False, Unused functions
    def build_eprim_tree(self, max_energy, slt_node, random_state, max_hop=None):
        edges = copy.deepcopy(self.edges)

        used_edges, max_childs, _ = self.decompose_tree(edges, slt_node, max_energy, True)

        edge_list = self.sort_by_bfs_order(used_edges)

        C = set()
        A = rset()
        C.add(0)

        for i in range(self.number_of_vertices):
            if max_childs[i] != -float('inf'):
                C.add(i)
            elif i <= self.n and max_childs[i] == -float('inf'):
                C.add(i)
                d = distance(self._points[0], self._points[i])
                max_childs[i] = self.max_childs(i, max_energy, d, strict_lower=True)
            else:
                max_childs[i] = 0

        for u in C:
            if max_childs[u] > 0:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.initialize()
        for u, v in edge_list:
            self.add_edge(u, v)

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, strict_lower=True, max_hop=max_hop)

        if len(C) < self.node_count:
            for u in C:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, used_max_childs=False, strict_lower=True, max_hop=None)
        # print(len(self.edges))
        self.repair()

    def build_fprim_tree(self, max_energy, slt_node, random_state, max_hop=None):
        C = set()
        A = rset()
        C.add(0)

        max_childs = [0] * self.number_of_vertices 
        for i in range(1, self.n+1):
                C.add(i)
                d = distance(self._points[0], self._points[i])
                max_childs[i] = self.max_childs(i, max_energy, d, strict_lower=True)

        for u in C:
            if max_childs[u] > 0:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.initialize()

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, strict_lower=True, max_hop=max_hop)

        if len(C) < self.node_count:
            for u in C:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, used_max_childs=False, strict_lower=True, max_hop=max_hop)
        # print(len(self.edges))

        if len(C) < self.node_count:
            for u in C:
                for v in self.potential_adj[u]:
                    if v not in C:
                        A.add((u, v))

        self.run_algorithm(C, A, self.potential_adj, max_childs, max_energy, random_state, used_max_childs=False, strict_lower=True, max_hop=None)

        self.repair()
 
        
