# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
CLRS30 Taxonomy

Greedy: ["Jet Lag - H"] 

Graphs: ["Turning Red - G", "Alea Lacta Est - K"]

Dynamic Programming: ["A Recurring Problem - E", "Bridging the Gap - J"]

Geometry: ["Three Kinds of Dice - C", "Carl's Vacation - D", "Waterworld - I"]

Strings: ["Tilting Tiles - F"]

Search: ["Schedule - B"]

???: ["Riddle of the Sphinx - A"]
"""

"""Algorithm specs.

The "spec" of each algorithm is a static set of `(stage, loc, type)`-tuples.

- `stage`: One of either an `input`, `output` or `hint`
- `location`: Each datum is associated with either the `node`, `edge` or `graph`
- `type`: Either a `scalar`, `categorical`, `mask`, `mask_one` or `pointer`

The dataflow for an algorithm is represented by `(stage, loc, type, data)`
"probes" that are valid under that algorithm's spec. It contains a single
snapshot for each `input` and `output` and a time-series of intermediate
algorithmic states (`hint`).

At minimum, each node contains a `pos` probe that serves as a unique index e.g.
for representing sequential data where appropriate
"""

# Specs definition for each of the algorithms in the CLRS30 dataset.
# For the original implementation, please refer to: https://github.com/google-deepmind/clrs

# This implementation is pretty much the same as the original one, 
# but with some minor changes in the way the data is handled to better fit the Dataset in torch framework format.
# As the original implementation is made in the Haiku framework, which is a JAX-based framework,

# Modifications were made by Ronald Albert (https://www.pesc.coppe.ufrj.br/index.php/pt-BR/pessoas/details/18/2955), throughout the course of his Master's degree in Computer Science 
# at the Federal University of Rio de Janeiro (UFRJ), Brazil.

import types
from typing import Dict, Tuple


class Stage:
  INPUT = 'input'
  OUTPUT = 'output'
  HINT = 'hint'


class Location:
  NODE = 'node'
  EDGE = 'edge'
  GRAPH = 'graph'


class Type:
  SCALAR = 'scalar'
  CATEGORICAL = 'categorical'
  MASK = 'mask'
  MASK_ONE = 'mask_one'
  POINTER = 'pointer'
  SHOULD_BE_PERMUTATION = 'pointer'
  PERMUTATION_POINTER = 'permutation_pointer'
  SOFT_POINTER = 'soft_pointer'


class OutputClass:
  POSITIVE = 1
  NEGATIVE = 0
  MASKED = -1

Spec = Dict[str, Tuple[str, str, str]]

CLRS_30_ALGS = [
    'articulation_points',
    'activity_selector',
    'bellman_ford',
    'bfs',
    'binary_search',
    'bridges',
    'bubble_sort',
    'dag_shortest_paths',
    'dfs',
    'dijkstra',
    'find_maximum_subarray_kadane',
    'floyd_warshall',
    'graham_scan',
    'heapsort',
    'insertion_sort',
    'jarvis_march',
    'kmp_matcher',
    'lcs_length',
    'matrix_chain_order',
    'minimum',
    'mst_kruskal',
    'mst_prim',
    'naive_string_matcher',
    'optimal_bst',
    'quickselect',
    'quicksort',
    'segments_intersect',
    'strongly_connected_components',
    'task_scheduling',
    'topological_sort'
]


ALGO_IDX_INPUT_NAME = 'algo_idx'

# Algorithms have varying numbers of signals they are evaluated on.
# To compensate for that, we issue more samples for those who use a small
# number of signals.
CLRS_30_ALGS_SETTINGS = {alg: {'num_samples_multiplier': 1}
                         for alg in CLRS_30_ALGS}
CLRS_30_ALGS_SETTINGS['find_maximum_subarray_kadane'][
    'num_samples_multiplier'] = 32
for alg in ['quickselect', 'minimum', 'binary_search', 'naive_string_matcher',
            'kmp_matcher', 'segments_intersect']:
  CLRS_30_ALGS_SETTINGS[alg]['num_samples_multiplier'] = 64


SPECS = types.MappingProxyType({
    'insertion_sort': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.SHOULD_BE_PERMUTATION),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'bubble_sort': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.SHOULD_BE_PERMUTATION),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'heapsort': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'parent': (Stage.HINT, Location.NODE, Type.POINTER),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'largest': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'heap_size': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL)
    },
    'quicksort': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.SHOULD_BE_PERMUTATION),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'p': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'r': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'quickselect': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'median': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'p': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'r': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i_rank': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'target': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'pivot': (Stage.HINT, Location.NODE, Type.MASK_ONE), 
    },
    'minimum': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'min': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'min_h': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'binary_search': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'target': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        'return': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'mid': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'find_maximum_subarray': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'start': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'end': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'mid': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'left_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'left_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'left_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'right_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'right_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'right_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'cross_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'cross_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'cross_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'ret_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'ret_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'ret_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'left_x_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'right_x_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL)
    },
    'find_maximum_subarray_kadane': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'start': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'end': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'best_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'best_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'best_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'sum': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'matrix_chain_order': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'p': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.OUTPUT, Location.EDGE, Type.POINTER),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'm': (Stage.HINT, Location.EDGE, Type.SCALAR),
        's_h': (Stage.HINT, Location.EDGE, Type.POINTER),
        'msk': (Stage.HINT, Location.EDGE, Type.MASK)
    },
    'lcs_length': {
        'string': (Stage.INPUT, Location.NODE, Type.MASK),
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.CATEGORICAL),
        'b': (Stage.OUTPUT, Location.EDGE, Type.CATEGORICAL),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'b_h': (Stage.HINT, Location.EDGE, Type.CATEGORICAL),
        'c': (Stage.HINT, Location.EDGE, Type.SCALAR)
    },
    'optimal_bst': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'p': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'q': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'root': (Stage.OUTPUT, Location.EDGE, Type.POINTER),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'root_h': (Stage.HINT, Location.EDGE, Type.POINTER),
        'e': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'w': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'msk': (Stage.HINT, Location.EDGE, Type.MASK)
    },
    'activity_selector': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'f': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'selected': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'selected_h': (Stage.HINT, Location.NODE, Type.MASK),
        'm': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'task_scheduling': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'd': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'w': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'selected': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'selected_h': (Stage.HINT, Location.NODE, Type.MASK),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        't': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'dfs': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'topological_sort': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'topo': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'topo_head': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'topo_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'topo_head_h': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'strongly_connected_components': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'scc_id': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'scc_id_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'A_t': (Stage.HINT, Location.EDGE, Type.MASK),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK)
    },
    'articulation_points': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'is_cut': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'is_cut_h': (Stage.HINT, Location.NODE, Type.MASK),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        'low': (Stage.HINT, Location.NODE, Type.SCALAR),
        'child_cnt': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'bridges': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'is_bridge': (Stage.OUTPUT, Location.EDGE, Type.MASK),
        'is_bridge_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        'low': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'bfs': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'reach_h': (Stage.HINT, Location.NODE, Type.MASK),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER)
    },
    'mst_kruskal': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'in_mst': (Stage.OUTPUT, Location.EDGE, Type.MASK),
        'in_mst_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'pi': (Stage.HINT, Location.NODE, Type.POINTER),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'root_u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'root_v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'mask_u': (Stage.HINT, Location.NODE, Type.MASK),
        'mask_v': (Stage.HINT, Location.NODE, Type.MASK),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL)
    },
    'mst_prim': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'key': (Stage.HINT, Location.NODE, Type.SCALAR),
        'mark': (Stage.HINT, Location.NODE, Type.MASK),
        'in_queue': (Stage.HINT, Location.NODE, Type.MASK),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'bellman_ford': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'msk': (Stage.HINT, Location.NODE, Type.MASK)
    },
    'dag_shortest_paths': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'mark': (Stage.HINT, Location.NODE, Type.MASK),
        'topo_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'topo_head_h': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK)
    },
    'dijkstra': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'mark': (Stage.HINT, Location.NODE, Type.MASK),
        'in_queue': (Stage.HINT, Location.NODE, Type.MASK),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'floyd_warshall': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'Pi': (Stage.OUTPUT, Location.EDGE, Type.POINTER),
        'Pi_h': (Stage.HINT, Location.EDGE, Type.POINTER),
        'D': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'msk': (Stage.HINT, Location.EDGE, Type.MASK),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'bipartite_matching': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        't': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'in_matching': (Stage.OUTPUT, Location.EDGE, Type.MASK),
        'in_matching_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'A_h': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'adj_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'msk': (Stage.HINT, Location.NODE, Type.MASK),
        'pi': (Stage.HINT, Location.NODE, Type.POINTER),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK)
    },
    'naive_string_matcher': {
        'string': (Stage.INPUT, Location.NODE, Type.MASK),
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.CATEGORICAL),
        'match': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
    'kmp_matcher': {
        'string': (Stage.INPUT, Location.NODE, Type.MASK),
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.CATEGORICAL),
        'match': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'pi': (Stage.HINT, Location.NODE, Type.POINTER),
        'is_reset': (Stage.HINT, Location.NODE, Type.MASK),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'k_reset': (Stage.HINT, Location.GRAPH, Type.MASK),
        'q': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'q_reset': (Stage.HINT, Location.GRAPH, Type.MASK),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK)
    },
    'segments_intersect': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'x': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'y': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'intersect': (Stage.OUTPUT, Location.GRAPH, Type.MASK),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'dir': (Stage.HINT, Location.NODE, Type.SCALAR),
        'on_seg': (Stage.HINT, Location.NODE, Type.MASK)
    },
    'graham_scan': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'x': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'y': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'in_hull': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'best': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'atans': (Stage.HINT, Location.NODE, Type.SCALAR),
        'in_hull_h': (Stage.HINT, Location.NODE, Type.MASK),
        'stack_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        'last_stack': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL)
    },
    'jarvis_march': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'x': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'y': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'in_hull': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'in_hull_h': (Stage.HINT, Location.NODE, Type.MASK),
        'best': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'last_point': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'endpoint': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL)
    },
    'schedule': {
        'n': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        'w': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
        'c': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
        'infinity':(Stage.OUTPUT, Location.GRAPH, Type.MASK),
        'c_h': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    'three_kinds_dice': {
        "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
        'values_D1': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'values_D2': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'output_score_D1': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
        'output_score_D2': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
        'score_D1': (Stage.HINT, Location.NODE, Type.SCALAR),
        'score_D2': (Stage.HINT, Location.NODE, Type.SCALAR),
        "in_hull": (Stage.HINT, Location.NODE, Type.MASK)
    },
    'carls_vacation': {
        "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
        'x': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'y': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'height1': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        'height2': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        'distance': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
        'faces1_x': (Stage.HINT, Location.NODE, Type.SCALAR),
        'faces1_y': (Stage.HINT, Location.NODE, Type.SCALAR),
        'faces2_x': (Stage.HINT, Location.NODE, Type.SCALAR),
        'faces2_y': (Stage.HINT, Location.NODE, Type.SCALAR),
        'tops_segment1_x': (Stage.HINT, Location.NODE, Type.SCALAR),
        'tops_segment1_y': (Stage.HINT, Location.NODE, Type.SCALAR),
        'tops_segment2_x': (Stage.HINT, Location.NODE, Type.SCALAR),
        'tops_segment2_y': (Stage.HINT, Location.NODE, Type.SCALAR),
        'faces2_y': (Stage.HINT, Location.NODE, Type.SCALAR),
        'selected_segment1': (Stage.HINT, Location.NODE, Type.MASK),
        'selected_segment2': (Stage.HINT, Location.NODE, Type.MASK),
    },
    "waterworld": {
        "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
        'n': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        'm': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        'area_percentages': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'surface_percentage':(Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
        'total_area': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'area_sums': (Stage.HINT, Location.GRAPH, Type.SCALAR)
    },
    "jet_lag": {
        "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
        'b': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'e': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'N': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        'impossible': (Stage.OUTPUT, Location.GRAPH, Type.MASK),
    }
})

CATEGORIES_DIMENSIONS = {
    "heapsort": {
        "phase": 3
    },
    "lcs_length": {
        "key": 4,
        'b': 4,
        "b_h": 4
    },
    "dfs": {
        "color": 3
    },
    "topological_sort": {
        "color": 3
    },
    "strongly_connected_components": {
        "color": 3
    },
    "articulation_points": {
        "color": 3
    },
    "bridges": {
        "color": 3
    },
    "mst_kruskal" : {
        "phase": 3
    },
    "dag_shortest_paths": {
        "color": 3
    },
    "naive_string_matcher": {
        "key": 4
    },
    "kmp_matcher": {
        "key": 4
    },
    "graham_scan": {
        "phase": 5
    },
    "jarvis_march": {
        "phase": 2
    }
}
