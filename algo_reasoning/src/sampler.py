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
import torch
import torch.linalg as LA
from torch.utils.data import IterableDataset
import math
from typing import Any, Callable, Optional, Tuple, List

from algo_reasoning.src.data import collate

# Import algorithms
from algo_reasoning.src.algorithms.scheduleB import schedule
from algo_reasoning.src.algorithms.three_kinds_diceC import three_kinds_dice
from algo_reasoning.src.algorithms.carls_vacationD import carls_vacation
from algo_reasoning.src.algorithms.jet_lagH import jet_lag
from algo_reasoning.src.algorithms.waterworldI import waterworld

from algo_reasoning.src.algorithms.sorting import insertion_sort, bubble_sort, heapsort, quicksort
from algo_reasoning.src.algorithms.greedy import activity_selector, task_scheduling
from algo_reasoning.src.algorithms.dynamic_programming import matrix_chain_order, lcs_length, optimal_bst
from algo_reasoning.src.algorithms.searching import minimum, binary_search, quickselect
from algo_reasoning.src.algorithms.divide_and_conquer import find_maximum_subarray_kadane
from algo_reasoning.src.algorithms.strings import naive_string_matcher, kmp_matcher
from algo_reasoning.src.algorithms.geometry import segments_intersect, graham_scan, jarvis_march
from algo_reasoning.src.algorithms.graphs import dfs, bfs, topological_sort, articulation_points, bridges, strongly_connected_components, mst_kruskal, mst_prim, bellman_ford, dijkstra, dag_shortest_paths, floyd_warshall

Algorithm = Callable[..., Any]

# TODO: Implement test script for Sampler classes
class BaseAlgorithmSampler:
    """Sampler abstract base class."""

    def __init__(
        self,
        algorithm: Algorithm,
        generator: Optional[torch.Generator] = None,
        randomize_pos : bool = True
    ):
        """Initializes a `Sampler`.

        Args:
        algorithm: The algorithm to sample from
        seed: RNG seed.
        randomize_pos: Whether to randomize input position of nodes,
        *args: Algorithm args.
        **kwargs: Algorithm kwargs.
        """
        if generator is None:
            self._generator = torch.Generator()
        else:
            self._generator = generator
        self._algorithm = algorithm
        self.randomize_pos = randomize_pos

    def sample(self, nb_nodes: int, batch_size: int, *args, **kwargs):
        """Samples trajectories from the pre-generated dataset.
            Args:
            nb_nodes: Number of nodes in retrieved batch.
            batch_size: Number of samples to retrieve.

            Returns:
            Subsampled trajectories.
        """
        data_list = []

        for _ in range(batch_size):
            data = self._sample_data(nb_nodes, *args, **kwargs)
            if self.randomize_pos:
                random_pos_args = dict(
                    pos_generator=self._generator
                )

                clrs_data = self._algorithm(*data, **random_pos_args)
            else: 
                clrs_data = self._algorithm(*data) 
            data_list.append(clrs_data)

        # Batch and pad trajectories to max(T).
        batched_data = collate(data_list)
        return batched_data

    def _sample_data(self, nb_nodes: int, *args, **kwargs):
        pass

# ICPC Problems Samplers
class ScheduleSampler(BaseAlgorithmSampler):
    """Scedule - B sampler."""  
    def __init__(self, *args, **kwargs):
        algorithm = schedule
        super().__init__(algorithm, *args, **kwargs)
        
    def _sample_data(self,
        nb_nodes: int,
        n_min: int = 2,
        n_max: int =  10**2,
        w_min: int = 1,
        w_max: int = 53
    ):
        """Sample inputs."""
        n = torch.randint(n_min, n_max, (), generator=self._generator).item()
        w = torch.randint(w_min, w_max, (), generator=self._generator).item()

        return [n, w, nb_nodes]
  
class ThreeKindsDiceSampler(BaseAlgorithmSampler):
    """Three Kinds Dice - C Sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = three_kinds_dice
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        faces_min: int = 1,
        faces_max: int = 10**2
    ):
        N_faces1 = torch.randint(faces_min, faces_max, (), generator=self._generator).item()
        N_faces2 = torch.randint(faces_min, faces_max, (), generator=self._generator).item()

        values_D1 = torch.randint(1, nb_nodes, (N_faces1, ))
        values_D2 = torch.randint(1, nb_nodes, (N_faces2, ))
   
        return [values_D1, values_D2, nb_nodes]

class CarlsVacationSampler(BaseAlgorithmSampler):
    """Carl's Vacation - D Sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = carls_vacation
        super().__init__(algorithm, *args, **kwargs)

    def generate_non_intersecting_squares(self, max_value: int, max_distance: int, max_height: int):
        p3 = ((torch.rand((2)) * (max_distance + max_value)) - max_value)
        p4 = p3 + torch.rand((2)) * max_value

        distance = LA.vector_norm(p3 - p4).item()
        diameter = distance * math.sqrt(2)
        
        p1 = (p3 - diameter) - torch.rand((2)) * max_value
        p2 = p1 - torch.rand((2)) * max_value

        height1, height2 = ((torch.rand((2)) * (max_height + max_value)) - max_value)

        x = torch.tensor([p1[0], p2[0], p3[0], p4[0]])
        y = torch.tensor([p1[1], p2[1], p3[1], p4[1]])

        return x, y, height1.item(), height2.item()

    def _sample_data(self, 
        nb_nodes:int = 4,
        max_value: int = 10**2,
        max_distance: int = 10**3,
        max_height: int = 10**2
        ):
    
        assert nb_nodes == 4, "nb_nodes must be 4 for Carl's Vacation - D"

        x, y, height, height2 = self.generate_non_intersecting_squares(max_value, max_distance, max_height)
        
        return [x, y, height, height2, nb_nodes]

class JetLagSampler(BaseAlgorithmSampler):
    """Jet Lag - H Sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = jet_lag
        super().__init__(algorithm, *args, **kwargs)

    def sample_tasks(self, 
                    nb_nodes: int, 
                    activity_dur: int, 
                    interval: int):
        last_e = 0
        b = torch.tensor([])
        e = torch.tensor([])

        for _ in range(nb_nodes):
            last_b = last_e + torch.randint(interval, (1,), generator=self._generator)

            b = torch.concat((b, last_b), dim=0)
            e = torch.concat((e, last_b + torch.randint(activity_dur, (1,), generator=self._generator)), dim=0)

            last_e = e[-1].item()

        return b, e

    def _sample_data(self,
        nb_nodes,
        max_activity_dur: int = 100,
        max_interval: int = 100):

        activity_dur = torch.randint(5, max_activity_dur, (), generator=self._generator).item()
        interval = torch.randint(5, max_interval, (), generator=self._generator).item()

        b, e = self.sample_tasks(nb_nodes, activity_dur, interval)

        return [b, e, nb_nodes]

class WaterworldSampler(BaseAlgorithmSampler):
    """Waterworld - I Sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = waterworld
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self, nb_nodes: int):
        
        factors = []
        for i in range(1, int(nb_nodes**0.5) + 1):
            if nb_nodes % i == 0:
                factors.append((i, nb_nodes // i))

        idx = torch.randint(len(factors), (), generator=self._generator).item()
        n, m = factors[idx]

        ap = torch.randint(101, (nb_nodes,), generator=self._generator)

        return [n, m, ap, nb_nodes]
    
# Sorting Algorithms Samplers
class BaseSortingSampler(BaseAlgorithmSampler):
    """Sorting Algorithms Sampler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sample_data(self, 
                    nb_nodes: int,
                    low: float = 0.0,
                    high: float = 1.0):

        A = torch.rand((nb_nodes,), generator=self._generator) * (high - low) + low 

        return [A, nb_nodes]

class InsertionSortSampler(BaseSortingSampler):
    """Insertion Sort Sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = insertion_sort
        super().__init__(algorithm, *args, **kwargs)

class BubbleSortSampler(BaseSortingSampler):
    """Bubble Sort Sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = bubble_sort
        super().__init__(algorithm, *args, **kwargs)

class HeapSortSampler(BaseSortingSampler):
    """Heap Sort Sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = heapsort
        super().__init__(algorithm, *args, **kwargs)

class QuickSortSampler(BaseSortingSampler):
    """Quick Sort Sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = quicksort
        super().__init__(algorithm, *args, **kwargs)


# Greedy Algorithms Samplers
class ActivitySelectorSampler(BaseAlgorithmSampler):
    """Activity Selector Sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = activity_selector
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self, 
                    nb_nodes: int,
                    low: float = 0.,
                    high: float = 1.,):

        arr_1 = torch.rand((nb_nodes,), generator=self._generator) * (high - low) + low 
        arr_2 = torch.rand((nb_nodes,), generator=self._generator) * (high - low) + low

        return [torch.minimum(arr_1, arr_2), torch.maximum(arr_1, arr_2), nb_nodes]
    
class TaskSchedulingSampler(BaseAlgorithmSampler):
    def __init__(self, *args, **kwargs):
        algorithm = task_scheduling
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self, 
                    nb_nodes: int,
                    low: float = 0.,
                    high: float = 1.,):

        d = torch.randint(high=nb_nodes, size=(nb_nodes,), generator=self._generator) + 1
        w = torch.rand((nb_nodes,), generator=self._generator) * (high - low) + low

        return [d, w, nb_nodes]
    
# Dynamic Programming Algorithms Samplers
class MatrixChainOrderSampler(BaseSortingSampler):
    def __init__(self, *args, **kwargs):
        algorithm = matrix_chain_order
        super().__init__(algorithm, *args, **kwargs)

class LCSLengthSampler(BaseAlgorithmSampler):

    def __init__(self, *args, **kwargs):
        algorithm = lcs_length
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self, 
                    nb_nodes: int,
                    chars: int = 4):
        length = nb_nodes // 2
        length_2 = nb_nodes - length
        
        x = torch.randint(high=chars, size=(length,), generator=self._generator)
        y = torch.randint(high=chars, size=(length_2,), generator=self._generator)

        return [x, y, nb_nodes]
    
class OptimalBSTSampler(BaseAlgorithmSampler):
    def __init__(self, *args, **kwargs):
        algorithm = optimal_bst
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self, 
                    nb_nodes: int):
        length = nb_nodes - 1 

        tot_length = length + (length + 1)
        arr = torch.rand((tot_length,), generator=self._generator)
        arr /= torch.sum(arr)
        
        p = arr[:length]
        q = arr[length:]

        return [p, q, nb_nodes]
    
# Searching Algorithms Samplers
class MinimumSampler(BaseSortingSampler):
    def __init__(self, *args, **kwargs):
        algorithm = minimum
        super().__init__(algorithm, *args, **kwargs)

class BinarySearchSampler(BaseAlgorithmSampler):
    def __init__(self, *args, **kwargs):
        algorithm = binary_search
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self, 
                    nb_nodes: int,
                    low: float = 0.,
                    high: float = 1.,):

        A = torch.rand((nb_nodes,), generator=self._generator) * (high - low) + low 
        x = torch.rand((1,), generator=self._generator) * (high - low) + low

        return [x.item(), A, nb_nodes]

class QuickselectSampler(BaseSortingSampler):
    def __init__(self, *args, **kwargs):
        algorithm = quickselect
        super().__init__(algorithm, *args, **kwargs)

# Divide and Conquer Algorithms Samplers

class MaximumSubarraySampler(BaseSortingSampler):
    def __init__(self, *args, **kwargs):
        algorithm = find_maximum_subarray_kadane
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self, 
                    nb_nodes: int,
                    low: float = -1.0,
                    high: float = 1.0):
        
        return super()._sample_data(nb_nodes, low, high)
    
class StringsSampler(BaseAlgorithmSampler):
    """String Algorithms Sampler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,  
        length_needle: Optional[int] = -8,
        chars: int = 4):

        if length_needle is None:
            if nb_nodes < 5:
                length_needle = 1
            else:
                length_needle = nb_nodes // 5
        elif length_needle < 0:
            length_needle = torch.randint(low=1, high=1 - length_needle, size=()).item()

        length_haystack = nb_nodes - length_needle
        needle = torch.randint(high=chars, size=(length_needle,), generator=self._generator)
        haystack = torch.randint(high=chars, size=(length_haystack,), generator=self._generator)
        embed_pos = torch.randint(high=(length_haystack - length_needle), size=(), generator=self._generator).item()

        haystack[embed_pos:embed_pos + length_needle] = needle
        return [haystack, needle, nb_nodes]
    
class NaiveStringMatcherSampler(StringsSampler):
    """Naive String Matcher Sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = naive_string_matcher
        super().__init__(algorithm, *args, **kwargs)

class KMPMatcherSampler(StringsSampler):
    """KMP MatcherSampler."""
    def __init__(self, *args, **kwargs):
        algorithm = kmp_matcher
        super().__init__(algorithm, *args, **kwargs)


# Geometry Algorithms Samplers
class GeometrySampler(BaseAlgorithmSampler):
    """Geometry Algorithms Sampler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sample_data_segments(self, 
                            nb_nodes: int, 
                            low: float = 0.,
                            high: float = 1.,):
        assert nb_nodes == 4, "nb_nodes must be 4 for Segments Intersect"

        def ccw(x_a, y_a, x_b, y_b, x_c, y_c):
            return ((y_c - y_a) * (x_b - x_a) > (y_b - y_a) * (x_c - x_a)).item() 
                    
        def intersect(xs, ys):
            return ccw(xs[0], ys[0], xs[2], ys[2], xs[3], ys[3]) != ccw(xs[1], ys[1], xs[2], ys[2], xs[3], ys[3]) and (
                ccw(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]) != ccw(xs[0], ys[0], xs[1], ys[1], xs[3], ys[3]))
        
        _intersect = torch.bernoulli(torch.tensor(0.5), generator=self._generator).item()
        
        xs = torch.rand((nb_nodes,), generator=self._generator) * (high - low) + low 
        ys = torch.rand((nb_nodes,), generator=self._generator) * (high - low) + low 

        while intersect(xs, ys) != _intersect:
            xs = torch.rand((nb_nodes,), generator=self._generator) * (high - low) + low 
            ys = torch.rand((nb_nodes,), generator=self._generator) * (high - low) + low 

        return [xs, ys, nb_nodes]

    def _sample_data_convex_hull(self, 
                    nb_nodes: int,
                    origin_x: float = 0.,
                    origin_y: float = 0., 
                    radius: float = 2.):
        low = 0.0
        high = 2 * torch.pi

        thetas = torch.rand((nb_nodes,), generator=self._generator) * (high - low) + low 
        rs = radius * torch.sqrt(torch.rand((nb_nodes,), generator=self._generator))

        xs = rs * torch.cos(thetas) + origin_x
        ys = rs * torch.sin(thetas) + origin_y

        return [xs, ys, nb_nodes]
    
class SegmentsIntersectSampler(GeometrySampler):
    def __init__(self, *args, **kwargs):
        algorithm = segments_intersect
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self, 
                    nb_nodes: int, 
                    low: float = 0.,
                    high: float = 1.,):
        
        return super()._sample_data_segments(nb_nodes, low, high)

class GrahamScanSampler(GeometrySampler):
    def __init__(self, *args, **kwargs):
        algorithm = graham_scan
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self, 
                    nb_nodes: int, 
                    origin_x: float = 0.,
                    origin_y: float = 0., 
                    radius: float = 2.):
        
        return super()._sample_data_convex_hull(nb_nodes, origin_x, origin_y, radius)
    
class JarvisMarchSampler(GeometrySampler):
    def __init__(self, *args, **kwargs):
        algorithm = jarvis_march
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self, 
                    nb_nodes: int, 
                    origin_x: float = 0.,
                    origin_y: float = 0., 
                    radius: float = 2.):
        
        return super()._sample_data_convex_hull(nb_nodes, origin_x, origin_y, radius)

class GraphSampler(BaseAlgorithmSampler):
    """Graph Algorithms Sampler."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sample_erdos_renyi(self, nb_nodes, 
                        p=0.5, 
                        directed=False, 
                        acyclic=False,
                        weighted=False, 
                        low=0.0, 
                        high=1.0):
        """Random Erdos-Renyi graph."""
        mat = torch.bernoulli(p * torch.ones(nb_nodes, nb_nodes), generator=self._generator)

        if not directed:
            mat *= torch.transpose(mat.clone(), 0, 1)
        elif acyclic:
            mat = torch.triu(mat.clone(), diagonal=1)
            p = torch.randperm(nb_nodes, generator=self._generator)
            mat = mat[p, :][:, p]

        if weighted:
            weights = torch.rand((nb_nodes, nb_nodes), generator=self._generator) * (high - low) + low 
            if not directed:
                weights *= torch.transpose(weights.clone(), 0, 1)
                weights = torch.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
            mat = mat.clone().float() * weights
        return mat
    
    def _random_community_graph(self, 
                            nb_nodes,
                            k=4, 
                            p=0.5, 
                            eps=0.01,
                            directed=False,
                            acyclic=False, 
                            weighted=False,
                            low=0.0, high=1.0):
        """Random perturbed k-community graph."""
        mat = torch.zeros((nb_nodes, nb_nodes))
        
        
        los, his = [], []
        lo = 0
        for i in range(k):
            if i == k - 1:
                hi = nb_nodes
            else:
                hi = lo + nb_nodes // k
            mat[lo:hi, lo:hi] = self._sample_erdos_renyi(
                hi - lo, p=p, directed=directed,
                acyclic=acyclic, weighted=weighted,
                low=low, high=high)
            
            los.append(lo)
            his.append(hi)
            lo = hi
        toggle = self._sample_erdos_renyi(nb_nodes, p=eps, directed=directed,
                                    acyclic=acyclic, weighted=weighted,
                                    low=low, high=high)

        # Prohibit closing new cycles
        for i in range(k):
            for j in range(i):
                toggle[los[i]:his[i], los[j]:his[j]] *= 0

        mat = torch.where(toggle > 0.0, (1.0 - (mat > 0.0).float()) * toggle, mat)
        p = torch.randperm(nb_nodes, generator=self._generator)
        mat = mat[p, :][:, p]
        return mat
    

class DFSSampler(GraphSampler):
    """DFS sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = dfs
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        p:Tuple[float, ...] = (0.5,)):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()

        graph = self._sample_erdos_renyi(nb_nodes, p=p[choice_p],
            directed=True, acyclic=False, weighted=False)
        
        return [graph, nb_nodes]
    
class BFSSampler(GraphSampler):
    """BFS sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = bfs
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        p:Tuple[float, ...] = (0.5,)):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()
        
        graph = self._sample_erdos_renyi(nb_nodes, p=p[choice_p],
            directed=False, acyclic=False, weighted=False)
        
        source_node = torch.randperm(nb_nodes, generator=self._generator)[0].item()

        return [graph, source_node, nb_nodes]
    
class TopologicalSortSampler(GraphSampler):
    """Topological Sort sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = topological_sort
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        p:Tuple[float, ...] = (0.5,)):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()

        graph = self._sample_erdos_renyi(nb_nodes, p=p[choice_p],
            directed=True, acyclic=True, weighted=False)
        
        return [graph, nb_nodes]
    
class ArticulationPointsSampler(GraphSampler):
    """Articulation Points sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = articulation_points
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        p:Tuple[float, ...] = (0.5,)):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()

        graph = self._sample_erdos_renyi(nb_nodes, p=p[choice_p],
            directed=False, acyclic=False, weighted=False)
        
        return [graph, nb_nodes]
    
class BridgesSampler(GraphSampler):
    """Bridges sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = bridges
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        p:Tuple[float, ...] = (0.5,)):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()

        graph = self._sample_erdos_renyi(nb_nodes, p=p[choice_p],
            directed=False, acyclic=False, weighted=False)
        
        return [graph, nb_nodes]
    
class MSTKruskalSampler(GraphSampler):
    """MST Kruskal sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = mst_kruskal
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        p:Tuple[float, ...] = (0.5,)):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()

        graph = self._sample_erdos_renyi(nb_nodes, p=p[choice_p],
            directed=False, acyclic=False, weighted=True)
        
        return [graph, nb_nodes]
    
class MSTPrimSampler(GraphSampler):
    """MST Prim sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = mst_prim
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        p:Tuple[float, ...] = (0.5,),
        low: float = 0.0,
        high: float = 1.0):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()

        graph = self._sample_erdos_renyi(nb_nodes, p=p[choice_p],
            directed=False, acyclic=False, weighted=True,
            low=low, high=high)
        
        source_node = torch.randperm(nb_nodes, generator=self._generator)[0].item()

        return [graph, source_node, nb_nodes]
    
class BellmanFordSampler(GraphSampler):
    """Bellman-Ford sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = bellman_ford
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        p:Tuple[float, ...] = (0.5,),
        low: float = 0.0,
        high: float = 1.0):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()

        graph = self._sample_erdos_renyi(nb_nodes, p=p[choice_p],
            directed=False, acyclic=False, weighted=True,
            low=low, high=high)
        
        source_node = torch.randperm(nb_nodes, generator=self._generator)[0].item()

        return [graph, source_node, nb_nodes]
    
class DijsktraSampler(GraphSampler):
    """Dijkstra sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = dijkstra
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        p:Tuple[float, ...] = (0.5,),
        low: float = 0.0,
        high: float = 1.0):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()

        graph = self._sample_erdos_renyi(nb_nodes, p=p[choice_p],
            directed=False, acyclic=False, weighted=True,
            low=low, high=high)
        
        source_node = torch.randperm(nb_nodes, generator=self._generator)[0].item()

        return [graph, source_node, nb_nodes]
    
class DAGShortestPathsSampler(GraphSampler):
    """DAG Shortest Paths sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = dag_shortest_paths
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        p:Tuple[float, ...] = (0.5,),
        low: float = 0.0,
        high: float = 1.0):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()

        graph = self._sample_erdos_renyi(nb_nodes, p=p[choice_p],
            directed=True, acyclic=True, weighted=True,
            low=low, high=high)
        
        source_node = torch.randperm(nb_nodes, generator=self._generator)[0].item()

        return [graph, source_node, nb_nodes]
    
class FloydWarshallSampler(GraphSampler):
    """Floyd-Warshall sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = floyd_warshall
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        p:Tuple[float, ...] = (0.5,),
        low: float = 0.0,
        high: float = 1.0):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()

        graph = self._sample_erdos_renyi(nb_nodes, p=p[choice_p],
            directed=False, acyclic=False, weighted=True,
            low=low, high=high)
        
        return [graph, nb_nodes]
    
class StronglyConnectedComponentsSampler(GraphSampler):
    """Strongly Connected Components sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = strongly_connected_components
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(self,
        nb_nodes: int,
        k: int = 4,
        p: Tuple[float, ...] = (0.5,),
        eps: float = 0.01):
        
        choice_p = torch.randperm(len(p), generator=self._generator)[0].item()

        graph = self._random_community_graph(
            nb_nodes=nb_nodes, k=k, p=p[choice_p], eps=eps,
            directed=True, acyclic=False, weighted=False)
        
        return [graph, nb_nodes]
    
SAMPLERS = {
    'articulation_points': ArticulationPointsSampler,
    'activity_selector': ActivitySelectorSampler,
    'bellman_ford': BellmanFordSampler,
    'bfs': BFSSampler,
    'binary_search': BinarySearchSampler,
    'bridges': BridgesSampler,
    'bubble_sort': BubbleSortSampler,
    'dag_shortest_paths': DAGShortestPathsSampler,
    'dfs': DFSSampler,
    'dijkstra': DijsktraSampler,
    'find_maximum_subarray_kadane': MaximumSubarraySampler,
    'floyd_warshall': FloydWarshallSampler,
    'graham_scan': GrahamScanSampler,
    'heapsort': HeapSortSampler,
    'insertion_sort': InsertionSortSampler,
    'jarvis_march': JarvisMarchSampler,
    'kmp_matcher': KMPMatcherSampler,
    'lcs_length': LCSLengthSampler,
    'matrix_chain_order': MatrixChainOrderSampler,
    'minimum': MinimumSampler,
    'mst_kruskal': MSTKruskalSampler,
    'mst_prim': MSTPrimSampler,
    'naive_string_matcher': NaiveStringMatcherSampler,
    'optimal_bst': OptimalBSTSampler,
    'quickselect': QuickselectSampler,
    'quicksort': QuickSortSampler,
    'segments_intersect': SegmentsIntersectSampler,
    'strongly_connected_components': StronglyConnectedComponentsSampler,
    'task_scheduling': TaskSchedulingSampler,
    'topological_sort': TopologicalSortSampler
}

class CLRSDataset(IterableDataset):
    def __init__(self,
                algorithms: List[str],
                nb_nodes: List[int],
                batch_size: int,
                num_samples: str,
                seed: Optional[int] = None,
                randomize_pos: bool = False,
                string_length: int = 20,
                algorithms_args: Optional[dict] = None):
        
        self.algorithms = algorithms
        self.seed = seed
        self._generator = torch.Generator().manual_seed(self.seed) if self.seed is not None else torch.Generator()
        self.nb_nodes = nb_nodes
        self.batch_size = batch_size
        self.string_length = string_length
        self.algorithms_args = algorithms_args
        self.num_samples = num_samples

        self.samplers = dict()
        for algo in algorithms:
            self.samplers[algo] = SAMPLERS[algo](generator=self._generator, randomize_pos=randomize_pos)

    def reset_generator(self, seed:int):
        self._generator.manual_seed(seed)
        

    def sample_data(self):
        for _ in range(len(self.algorithms) * self.num_samples):
            # Sample random algorithm and number of nodes
            algo = self.algorithms[torch.randint(len(self.algorithms), (), generator=self._generator).item()]

            if algo in ["segments_intersect", "carls_vacation"]:
                nb_nodes = 4
            elif algo in ["naive_string_matcher", "kmp_matcher"]:
                nb_nodes = self.string_length
            else:
                nb_nodes = self.nb_nodes[torch.randint(len(self.nb_nodes), (), generator=self._generator).item()]

            kwargs = self.algorithms_args[algo] if self.algorithms_args is not None and algo in self.algorithms_args else dict()

            sampled_data = self.samplers[algo].sample(nb_nodes, self.batch_size, **kwargs)

            yield sampled_data 

    def __iter__(self):
        return self.sample_data()