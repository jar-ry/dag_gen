"""DAG Graph Generator.
Generates a DAG
Author : Jarry Chen

.. MIT License
..
.. Copyright (c) 2023 Jarry Chen
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""

import numpy as np
import networkx as nx

class DAG_Generator:
    def __init__(self, nodes, parents_max, confounders):
        self.nodes = nodes
        self.confounders = confounders
        self.parents_max = parents_max
        self.adjacency_matrix = np.zeros((nodes, nodes))
        self.generate_count = 0
        self.nodes_with_two_or_more_parents = 0
        self.found_parent_to_remove = 0

    def generate(self):
        while True:
            self.generate_count += 1
            print(self.generate_count)
            # Initialize the adjacency matrix with edges
            for j in range(1, self.nodes):
                if self.confounders != 0 and self.parents_max == 1:
                    nb_parents = np.random.randint(
                    max(0, int(j * self.parents_max) - 1),
                    min(j, int(j * self.parents_max) + 1)
                )
                else:
                    nb_parents = np.random.randint(
                        0, min([self.parents_max, j]) + 1)
                for i in np.random.choice(range(0, j), nb_parents, replace=False):
                    self.adjacency_matrix[i, j] = 1

            # Ensure the graph is acyclic
            G = nx.DiGraph(self.adjacency_matrix)
            if not nx.is_directed_acyclic_graph(G):
                self.adjacency_matrix = np.zeros((self.nodes, self.nodes))
                print("graph not dag")
                continue
            
            # Connect disconnected nodes to a random node in the graph
            G = nx.DiGraph(self.adjacency_matrix)
            for node in G.nodes:
                if not G.in_edges(node) and not G.out_edges(node):
                    neighbors = list(G.nodes)
                    neighbors.remove(node)
                    random_neighbor = np.random.choice(neighbors)
                    self.adjacency_matrix[node, random_neighbor] = 1

            # Ensure at least one node has two parents when parents_max is 1
            if self.confounders:
                # Find a parent to remove for each node with two or more parents
                nodes_with_two_or_more_parents = np.where(np.sum(self.adjacency_matrix, axis=0) >= 2)[0]

                new_nodes_with_two_or_more_parents = []
                for parent in nodes_with_two_or_more_parents:
                    if nx.is_weakly_connected(G.subgraph(set(G.nodes)-set([parent]))):
                        new_nodes_with_two_or_more_parents.append(parent)
                nodes_with_two_or_more_parents = new_nodes_with_two_or_more_parents
                
                self.number_nodes_with_two_or_more_parents = len(nodes_with_two_or_more_parents)

                if self.number_nodes_with_two_or_more_parents < self.confounders:
                    print("number_nodes_with_two_or_more_parents less than confounders")
                    nodes_with_one_parent = np.where(np.sum(self.adjacency_matrix, axis=0) == 1)[0]
                    if len(nodes_with_one_parent) == 0:
                        self.adjacency_matrix = np.zeros((self.nodes, self.nodes))
                    node = np.random.choice(nodes_with_one_parent)
                    parents = np.where(self.adjacency_matrix[:, node] == 1)[0]
                    candidates = [i for i in range(0, self.nodes) if self.adjacency_matrix[i, node] == 0 and i not in parents]
                    if len(candidates) == 0:
                        self.adjacency_matrix = np.zeros((self.nodes, self.nodes))
                    for parent in candidates:
                        if nx.is_weakly_connected(G.subgraph(set(G.nodes)-set([parent]))):
                            self.number_nodes_with_two_or_more_parents += 1
                            self.adjacency_matrix[parent, node] = 1
                            if self.number_nodes_with_two_or_more_parents == self.confounders:
                                break

                nodes_with_two_or_more_parents = np.where(np.sum(self.adjacency_matrix, axis=0) >= 2)[0]
                final_check_list = []
                for parent in nodes_with_two_or_more_parents:
                    if nx.is_weakly_connected(G.subgraph(set(G.nodes)-set([parent]))):
                        final_check_list.append(parent)

                # If no parent can be removed for any node with two or more parents, discard the adjacency matrix and generate a new one
                if len(final_check_list) < self.confounders:
                    self.adjacency_matrix = np.zeros((self.nodes, self.nodes))
                    print("HERE4")
                    continue
    
            # Check if the graph is acyclic
            G = nx.DiGraph(self.adjacency_matrix)
            if nx.is_directed_acyclic_graph(G):
                return self.adjacency_matrix
            else:
                print("HERE5")
                self.adjacency_matrix = np.zeros((self.nodes, self.nodes))