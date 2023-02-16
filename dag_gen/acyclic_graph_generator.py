"""Acyclic Graph Generator.

Generates a dataset out of an acyclic FCM.
Author : Jarry Chen

.. MIT License
..
.. Copyright (c) 2022 Jarry Chen
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

from typing import Dict
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph

from .causal_mechanisms import (
    LinearMechanism,
    Polynomial_Mechanism,
    SigmoidAM_Mechanism,
    SigmoidMix_Mechanism,
    GaussianProcessAdd_Mechanism,
    GaussianProcessMix_Mechanism,
    NN_Mechanism,
    gmm_cause,
    normal_noise,
    uniform_noise,
)
import random
import copy
import json
import os

# import required modules
import inspect
import collections


class Regenerate_Dag(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        random_seed -- current random seed
        message -- explanation of the error
    """

    def __init__(self, random_seed, message):
        self.random_seed = random_seed
        self.message = message
        super().__init__(self.message)


class AcyclicGraphGenerator(object):
    """Generate an acyclic graph and data given a causal mechanism.

    Args:
        causal_mechanism (str): currently implemented mechanisms:
            ['linear', 'polynomial', 'sigmoid_add',
            'sigmoid_mix', 'gp_add', 'gp_mix', 'nn'].
        noise (str or function): type of noise to use in the generative process
            ('gaussian', 'uniform' or a custom noise function).
        noise_coeff (float): Proportion of noise in the mechanisms.
        initial_variable_generator (function): Function used to init variables
            of the graph, defaults to a Gaussian Mixture model.
        npoints (int): Number of data points to generate.
        nodes (int): Number of nodes in the graph to generate.
        parents_max (int): Maximum number of parents of a node.
        expected_degree (int): Degree (number of edge per node) expected,
            only used for erdos graph
        dag_type (str): type of graph to generate ('default', 'erdos')

    Example:
        >>> from cdt.data import AcyclicGraphGenerator
        >>> generator = AcyclicGraphGenerator('linear', npoints=1000)
        >>> data, graph = generator.generate()
        >>> generator.to_csv('generated_graph')
    """

    def __init__(
        self,
        causal_mechanism,
        noise="gaussian",
        noise_coeff=0.3,
        unfaithful_noise_coeff=0.2,
        selection_bias_nodes=0,
        selection_bias_ratio=0.5,
        selection_bias_range=[0, 0.5],
        initial_variable_generator=gmm_cause,
        npoints=500,
        nodes=20,
        parents_max=5,
        expected_degree=2,
        confounders=0,
        num_unfaithful_nodes=0,
        random_seed=42,
        dag_type="erdos",
    ):
        super(AcyclicGraphGenerator, self).__init__()

        # Set random seed so tests are reproducible
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Set causal mechisms
        causal_mechanism_list = []
        causal_mechanism_dict = {
            "linear": LinearMechanism,
            "polynomial": Polynomial_Mechanism,
            "sigmoid_add": SigmoidAM_Mechanism,
            "sigmoid_mix": SigmoidMix_Mechanism,
            "gp_add": GaussianProcessAdd_Mechanism,
            "gp_mix": GaussianProcessMix_Mechanism,
            "nn": NN_Mechanism,
        }
        node_parent_casual_mechanism = dict()
        nodes = nodes + confounders - num_unfaithful_nodes
        for i in range(1, nodes):
            if causal_mechanism == "mix":
                key = random.sample(causal_mechanism_dict.keys(), 1)[0]
                causal_mechanism_list.append(causal_mechanism_dict[key])
                node_parent_casual_mechanism[f"node_{i}"] = key
            else:
                causal_mechanism_list.append(causal_mechanism_dict[causal_mechanism])
                node_parent_casual_mechanism[f"node_{i}"] = causal_mechanism

        self.mechanism_list = causal_mechanism_list
        self.node_parent_mechanism = node_parent_casual_mechanism

        # Init data
        self.data = pd.DataFrame(None, columns=["V{}".format(i) for i in range(nodes)])

        # Set n samples to generate
        if selection_bias_nodes:
            self.n_to_remove = int(
                selection_bias_ratio
                * (
                    (selection_bias_range[1] * npoints)
                    - (selection_bias_range[0] * npoints)
                )
            )
            npoints = npoints + self.n_to_remove
        else:
            npoints = npoints
        self.npoints = int(npoints)

        # Set noise
        try:
            self.noise = {"gaussian": normal_noise, "uniform": uniform_noise}[noise]
        except KeyError:
            self.noise = noise

        # Set other variables
        self.confounders = confounders
        self.biased_nodes = None
        self.selection_bias_nodes = selection_bias_nodes
        self.selection_bias_ratio = selection_bias_ratio
        self.selection_bias_range = selection_bias_range
        self.num_unfaithful_nodes = num_unfaithful_nodes
        self.nodes = nodes
        self.unfaithful_noise_coeff = unfaithful_noise_coeff
        self.noise_coeff = noise_coeff
        self.adjacency_matrix = np.zeros((nodes, nodes))
        self.parents_max = parents_max
        self.expected_degree = expected_degree
        self.dag_type = dag_type
        self.initial_generator = initial_variable_generator
        self.cfunctions = None
        self.g = None
        self.confounder_data = None
        self.confounder_graph = None
        self.unfaithful_nodes = []
        self.deleted_nodes = []
        self.unfaithful_nodes_parents = []

    def init_dag(self, verbose):
        """Redefine the structure of the graph depending on dag_type
        ('default', 'erdos')

        Args:
            verbose (bool): Verbosity
        """
        if self.dag_type == "default":
            for j in range(1, self.nodes):
                nb_parents = np.random.randint(0, min([self.parents_max, j]) + 1)
                for i in np.random.choice(range(0, j), nb_parents, replace=False):
                    self.adjacency_matrix[i, j] = 1

        elif self.dag_type == "erdos":
            nb_edges = self.expected_degree * self.nodes
            prob_connection = 2 * nb_edges / (self.nodes ** 2 - self.nodes)
            causal_order = np.random.permutation(np.arange(self.nodes))

            for i in range(self.nodes - 1):
                node = causal_order[i]
                possible_parents = causal_order[(i + 1) :]
                num_parents = np.random.binomial(
                    n=self.nodes - i - 1, p=prob_connection
                )
                parents = np.random.choice(
                    possible_parents, size=num_parents, replace=False
                )
                self.adjacency_matrix[parents, node] = 1

        try:
            self.g = nx.DiGraph(self.adjacency_matrix)
            assert not list(nx.simple_cycles(self.g))

        except AssertionError:
            if verbose:
                print("Regenerating, graph non valid...")
            self.init_dag(verbose=verbose)

    def init_variables(self, verbose=False):
        """Redefine the causes, mechanisms and the structure of the graph,
        called by ``self.generate()`` if never called.

        Args:
            verbose (bool): Verbosity
        """
        self.init_dag(verbose)

        # Mechanisms
        self.cfunctions = [
            self.mechanism_list[i - 1](
                int(sum(self.adjacency_matrix[:, i])),
                self.npoints,
                self.noise,
                noise_coeff=self.noise_coeff,
                unfaithful_noise_coeff=self.unfaithful_noise_coeff,
            )
            if sum(self.adjacency_matrix[:, i])
            else self.initial_generator
            for i in range(self.nodes)
        ]

    def generate(self, rescale=True):
        """Generate data from an FCM defined in ``self.init_variables()``.

        Args:
            rescale (bool): rescale the generated data (recommended)

        Returns:
            tuple: (pandas.DataFrame, networkx.DiGraph, dict, pandas.DataFrame, networkx.DiGraph), respectively the
            generated data, graph, causal mechanisms, confounder data, confounder graph
        """
        if self.cfunctions is None:
            self.init_variables()
        curr_unfaithful_nodes = 0
        for i in nx.topological_sort(self.g):
            # Root cause
            if not sum(self.adjacency_matrix[:, i]):
                data = self.cfunctions[i](self.npoints)
                self.data[f"V{i}"] = data
            # Generating causes
            else:
                data, unfaithful_data = self.cfunctions[i](
                    self.data.iloc[:, self.adjacency_matrix[:, i].nonzero()[0]].values
                )
                self.data[f"V{i}"] = data

                # If unfaithful graph then save unfaithful data
                if curr_unfaithful_nodes < self.num_unfaithful_nodes:
                    self.data[f"UF_V{i}"] = unfaithful_data
                    self.unfaithful_nodes.append(i)
                    self.unfaithful_nodes_parents = list(
                        set(
                            self.unfaithful_nodes_parents
                            + [
                                i
                                for i, e in enumerate(self.adjacency_matrix[:, i])
                                if e == 1
                            ]
                        )
                    )
                    curr_unfaithful_nodes += 1
            if rescale:
                self.data[f"V{i}"] = scale(self.data[f"V{i}"].values)

        # Remove confounders
        if self.confounders:
            self.confounder_data = copy.deepcopy(self.data)
            self.confounder_graph = copy.deepcopy(self.g)
            for i in range(0, self.confounders):
                current_nodes = list(self.g.nodes)
                confounder_deleted = False
                while len(current_nodes) > 0 and not confounder_deleted:
                    random.shuffle(current_nodes)
                    random_sample = current_nodes.index(current_nodes[-1:][0])
                    del current_nodes[-1:]
                    delete_random_sample = False
                    # Check if sample is not a leaf node (horizontally no 1s for this number)
                    horizontal_vect = [
                        i
                        for i, e in enumerate(self.adjacency_matrix[random_sample])
                        if e == 1
                    ]
                    if len(horizontal_vect) != 0:
                        # Check if sample will disconnect the DAG into two graphs
                        # Get the horizontal axis nodes and vertiacal axis node of the sample and see if any other node point to these nodes
                        for idx in horizontal_vect:
                            if (
                                (
                                    1 in self.adjacency_matrix[:, idx]
                                    or 1 in self.adjacency_matrix[idx, :]
                                )
                                and random_sample not in self.unfaithful_nodes_parents
                                and random_sample not in self.unfaithful_nodes
                            ):
                                delete_random_sample = True
                            else:
                                delete_random_sample = False
                                break
                    if delete_random_sample:
                        self.g.remove_node(random_sample)
                        self.adjacency_matrix = np.delete(
                            self.adjacency_matrix, random_sample, 0
                        )
                        self.adjacency_matrix = np.delete(
                            self.adjacency_matrix, random_sample, 1
                        )
                        self.data.drop(
                            self.data.columns[random_sample], axis=1, inplace=True
                        )
                        confounder_deleted = True
                        self.deleted_nodes.append(random_sample)
            if len(self.deleted_nodes) < self.confounders:
                raise Regenerate_Dag(self.random_seed, "Confounder not generated")

        # Create selection bias
        biased_nodes = []
        if self.selection_bias_nodes:
            sample_idx = [
                x
                for x in list(range(self.g.number_of_nodes()))
                if (x not in self.deleted_nodes)
            ]
            biased_nodes = random.sample(sample_idx, self.selection_bias_nodes)
            for i in biased_nodes:
                target_series = self.data[f"V{i}"].copy()
                target_range = target_series[
                    (
                        target_series.index
                        >= np.percentile(
                            target_series.index, 100 * self.selection_bias_range[0]
                        )
                    )
                    & (
                        target_series.index
                        < np.percentile(
                            target_series.index, 100 * self.selection_bias_range[1]
                        )
                    )
                ]
                drop_indices = np.random.choice(
                    target_range.index, self.n_to_remove, replace=False
                )
                self.data.drop(drop_indices, inplace=True)
                if self.confounders:
                    self.confounder_data.drop(drop_indices, inplace=True)

        # Generate unfaithful node to cancel cause
        if self.num_unfaithful_nodes:
            for i in self.unfaithful_nodes:
                self.g.add_node(f"_unfaithful{i}")
                self.g.add_edge(f"_unfaithful{i}", f"{i}", weight=1)
                if self.confounders:
                    self.confounder_graph.add_node(f"_unfaithful{i}")
                    self.confounder_graph.add_edge(f"_unfaithful{i}", f"{i}", weight=1)

        if self.confounders:
            return (
                self.data,
                nx.relabel_nodes(
                    self.g, {i: "V" + str(i) for i in self.g.nodes}, copy=True
                ),
                self.node_parent_mechanism,
                self.confounder_data,
                nx.relabel_nodes(
                    self.confounder_graph,
                    {i: "V" + str(i) for i in self.confounder_graph.nodes},
                    copy=True,
                ),
            )
        return (
            self.data,
            nx.relabel_nodes(
                self.g, {i: "V" + str(i) for i in self.g.nodes}, copy=True
            ),
            self.node_parent_mechanism,
            None,
            None,
        )

    def generate_to_folder(self, data_path, data_index, rescale=True):
        """Generate data from an FCM defined in ``self.init_variables()``.

        Args:
            rescale (bool): rescale the generated data (recommended)
            folder_path (str): path to save output

        Returns:
            tuple: (pandas.DataFrame, networkx.DiGraph, dict, pandas.DataFrame, networkx.DiGraph), respectively the
            generated data, graph, causal mechanisms, confounder data, confounder graph
        """
        data, graph, node_parent_mech, conf_data, conf_graph = self.generate(rescale)

        np.save(data_path + f"/data{data_index}.npy", data.to_numpy())
        np.save(data_path + f"/DAG{data_index}.npy", nx.to_numpy_matrix(graph))
        if self.confounders:
            np.save(
                data_path + f"/confounder_data{data_index}.npy", conf_data.to_numpy()
            )
            np.save(
                data_path + f"/confounder_DAG{data_index}.npy",
                nx.to_numpy_matrix(conf_graph),
            )

        with open(data_path + "/causal_mechanisms.json", "w", encoding="utf-8") as f:
            json.dump(node_parent_mech, f, ensure_ascii=False, indent=4)
