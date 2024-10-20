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
import math
from typing import Any, Callable, Optional

from algo_reasoning.src.data import CLRSData, collate

# Import algorithms
from algo_reasoning.src.algorithms.scheduleB import schedule
from algo_reasoning.src.algorithms.three_kinds_diceC import three_kinds_dice
from algo_reasoning.src.algorithms.carls_vacationD import carls_vacation
from algo_reasoning.src.algorithms.jet_lagH import jet_lag
from algo_reasoning.src.algorithms.waterworldI import waterworld

from algo_reasoning.src.algorithms.sorting import insertion_sort, bubble_sort, heapsort, quicksort
from algo_reasoning.src.algorithms.greedy import activity_selector, task_scheduling


Algorithm = Callable[..., Any]

class BaseAlgorithmSampler:
    """Sampler abstract base class."""

    def __init__(
        self,
        algorithm: Algorithm,
        seed: Optional[int] = None,
        randomize_pos : bool = True,
        *args,
        **kwargs,
    ):
        """Initializes a `Sampler`.

        Args:
        algorithm: The algorithm to sample from
        seed: RNG seed.
        randomize_pos: Whether to randomize input position of nodes,
        *args: Algorithm args.
        **kwargs: Algorithm kwargs.
        """

        self._generator = torch.Generator()
        self._algorithm = algorithm
        self.randomize_pos = randomize_pos
        self._args = args
        self._kwargs = kwargs

        if seed is not None:
            self._generator.manual_seed(seed)

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
    
        assert nb_nodes == 4, "nb_nodes must be 4 for Carl's Vacation - D algorithm"

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
                    high: int = 1.0):

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
class ActivitySelectionSampler(BaseAlgorithmSampler):
    """Activity Selection Sampler."""
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