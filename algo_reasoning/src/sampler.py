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


import numpy as np
import torch
from typing import Any, Callable, List, Optional, Tuple

from algo_reasoning.src.data import CLRSData, collate
from algo_reasoning.src.algorithms.scheduleB import schedule

Algorithm = Callable[..., Any]

def _idx_batched_data(idx: int, batched_data: CLRSData) -> CLRSData:
    """Get itens at idx for batched data."""
    inputs_dict = {k: v[idx] for k, v in batched_data.inputs.items()}
    inputs = CLRSData(**inputs_dict)

    outputs_dict = {k: v[idx] for k, v in batched_data.outputs.items()}
    outputs = CLRSData(**outputs_dict)

    hints_dict = {k: v[idx] for k, v in batched_data.hints.items()}
    hints = CLRSData(**hints_dict)

    algorithm = batched_data.algorithm
    length = batched_data.length[idx]
    max_length = torch.max(length).long().item()

    return CLRSData(
        algorithm=algorithm,
        inputs=inputs,
        outputs=outputs,
        hints=hints,
        max_length=max_length,
        length=length,
    )

class BaseAlgorithmSampler():
    """Sampler abstract base class."""

    def __init__(
        self,
        algorithm: Algorithm,
        num_samples: int,
        nb_nodes: int, 
        seed: Optional[int] = None,
        randomize_pos = False,
        *args,
        **kwargs,
    ):
        """Initializes a `Sampler`.

        Args:
        algorithm: The algorithm to sample from
        num_samples: Number of algorithm unrolls to sample. If positive, all the
            samples will be generated in the constructor, and at each call of the
            `next` method a batch will be randomly selected among them. If -1,
            samples are generated on the fly with each call to `next`.
        seed: RNG seed.
        *args: Algorithm args.
        **kwargs: Algorithm kwargs.
        """

        self._generator = torch.Generator()
        self._num_samples = num_samples
        self._algorithm = algorithm
        self.randomize_pos = randomize_pos
        self.nb_nodes = nb_nodes
        self._args = args
        self._kwargs = kwargs

        if seed is not None:
            self._generator.manual_seed(seed)

        if num_samples >= 0:
            self.clrs_data = self._make_batch(num_samples, algorithm, *args, **kwargs)

    def _make_batch(self, num_samples: int, algorithm: Algorithm, *args, **kwargs):
        """Generate a batch of data."""
        data_list = []

        for _ in range(num_samples):
            data = self._sample_data(nb_nodes=self.nb_nodes, *args, **kwargs)
            if self.randomize_pos:
                random_pos_args = dict(
                    pos_generator=self._generator
                )

                clrs_data = algorithm(*data, **random_pos_args)
            else: 
                clrs_data = algorithm(*data) 
            data_list.append(clrs_data)

        # Batch and pad trajectories to max(T).
        batched_data = collate(data_list)
        return batched_data

    def sample(self, batch_size: Optional[int] = None):
        """Samples trajectories from the pre-generated dataset.
        Args:
        batch_size: Optional batch size. If `None`, returns entire dataset.

        Returns:
        Subsampled trajectories.
        """

        if self._num_samples < 0:
            assert batch_size is not None, "Batch size must be provided when samples are generated on the fly"

        if batch_size:
            if self._num_samples < 0:
                batched_data = self._make_batch(
                    batch_size,
                    self._algorithm,
                    *self._args,
                    **self._kwargs,
                )
            else:
                indices = torch.randint(self._num_samples, (batch_size,), generator=self._generator)
                batched_data = _idx_batched_data(indices, self.clrs_data)
        else:
            batched_data = self.clrs_data

        return batched_data

    def _sample_data(self, nb_nodes: int, *args, **kwargs):
        pass

#   def _random_sequence(self, length, low=0.0, high=1.0):
#     """Random sequence."""
#     return self._rng.uniform(low=low, high=high, size=(length,))

#   def _random_string(self, length, chars=4):
#     """Random string."""
#     return self._rng.randint(0, high=chars, size=(length,))

#   def _random_er_graph(self, nb_nodes, p=0.5, directed=False, acyclic=False,
#                        weighted=False, low=0.0, high=1.0):
#     """Random Erdos-Renyi graph."""

#     mat = self._rng.binomial(1, p, size=(nb_nodes, nb_nodes))
#     if not directed:_sample_data
#       mat *= np.transpose(mat)
#     elif acyclic:
#       mat = np.triu(mat, k=1)
#       p = self._rng.permutation(nb_nodes)  # To allow nontrivial solutions
#       mat = mat[p, :][:, p]
#     if weighted:
#       weights = self._rng.uniform(low=low, high=high, size=(nb_nodes, nb_nodes))
#       if not directed:
#         weights *= np.transpose(weights)
#         weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
#       mat = mat.astype(float) * weights
#     return mat

#   def _random_community_graph(self, nb_nodes, k=4, p=0.5, eps=0.01,
#                               directed=False, acyclic=False, weighted=False,
#                               low=0.0, high=1.0):
#     """Random perturbed k-community graph."""
#     mat = np.zeros((nb_nodes, nb_nodes))
#     if k > nb_nodes:
#       raise ValueError(f'Cannot generate graph of too many ({k}) communities.')
#     los, his = [], []
#     lo = 0
#     for i in range(k):
#       if i == k - 1:
#         hi = nb_nodes
#       else:
#         hi = lo + nb_nodes // k
#       mat[lo:hi, lo:hi] = self._random_er_graph(
#           hi - lo, p=p, directed=directed,
#           acyclic=acyclic, weighted=weighted,
#           low=low, high=high)
#       los.append(lo)
#       his.append(hi)
#       lo = hi
#     toggle = self._random_er_graph(nb_nodes, p=eps, directed=directed,
#                                    acyclic=acyclic, weighted=weighted,
#                                    low=low, high=high)

#     # Prohibit closing new cycles
#     for i in range(k):
#       for j in range(i):
#         toggle[los[i]:his[i], los[j]:his[j]] *= 0

#     mat = np.where(toggle > 0.0, (1.0 - (mat > 0.0)) * toggle, mat)
#     p = self._rng.permutation(nb_nodes)  # To allow nontrivial solutions
#     mat = mat[p, :][:, p]
#     return mat

#   def _random_bipartite_graph(self, n, m, p=0.25):
#     """Random bipartite graph-based flow network."""
#     nb_nodes = n + m + 2
#     s = 0
#     t = n + m + 1
#     mat = np.zeros((nb_nodes, nb_nodes))
#     mat[s, 1:n+1] = 1.0  # supersource
#     mat[n+1:n+m+1, t] = 1.0  # supersink
#     mat[1:n+1, n+1:n+m+1] = self._rng.binomial(1, p, size=(n, m))
#     return mat

class ScheduleSampler(BaseAlgorithmSampler):
    """Scedule sampler."""    
    def __init__(self, *args, **kwargs):
        algorithm = schedule
        super().__init__(algorithm, *args, **kwargs)
        
    def _sample_data(
        self,
        nb_nodes: int,
        n_min = 2,
        n_max =  10**2,
        w_min = 1,
        w_max = 53
    ):
        """Sample inputs."""
        n = torch.randint(n_min, n_max, (), generator=self._generator).item()
        w = torch.randint(w_min, w_max, (), generator=self._generator).item()

        return [n, w, nb_nodes]
  
class ThreeKindsDiceSampler(BaseAlgorithmSampler):
    """Three Kinds Dice Sampler."""
    def __init__(self, *args, **kwargs):
        algorithm = schedule
        super().__init__(algorithm, *args, **kwargs)

    def _sample_data(
        self,
        nb_nodes: int,
        faces_min = 1,
        faces_max = 10**2
    ):
        N_faces1 = torch.randint(faces_min, faces_max, (), generator=self._generator).item()
        N_faces2 = torch.randint(faces_min, faces_max, (), generator=self._generator).item()

        values_D1 = torch.randint(1, nb_nodes, (N_faces1, ))
        values_D2 = torch.randint(1, nb_nodes, (N_faces2, ))
   
        return [values_D1, values_D2, nb_nodes]