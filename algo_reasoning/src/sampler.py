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


import abc
import numpy as np
import torch
from typing import Any, Callable, List, Optional, Tuple

from clrs._src.samplers import Sampler
from algo_reasoning.src.data import CLRSData, collate
from algo_reasoning.src.specs import Spec


Algorithm = Callable[..., Any]

class Sampler(abc.ABC):
    """Sampler abstract base class."""

    def __init__(
        self,
        algorithm: Algorithm,
        spec: Spec,
        num_samples: int,
        seed: Optional[int] = None,
        track_max_steps: bool = True,
        *args,
        **kwargs,
    ):
        """Initializes a `Sampler`.

        Args:
        algorithm: The algorithm to sample from
        spec: The algorithm spec.
        num_samples: Number of algorithm unrolls to sample. If positive, all the
            samples will be generated in the constructor, and at each call of the
            `next` method a batch will be randomly selected among them. If -1,
            samples are generated on the fly with each call to `next`.
        seed: RNG seed.
        *args: Algorithm args.
        **kwargs: Algorithm kwargs.
        """

        self._generator = torch.Generator()
        self._spec = spec
        self._num_samples = num_samples
        self._algorithm = algorithm
        self._args = args
        self._kwargs = kwargs
        self._track_max_steps = track_max_steps

        if num_samples >= 0:
            self.clrs_data = self._make_batch(num_samples, algorithm, *args, **kwargs)

    def _make_batch(self, num_samples: int, algorithm: Algorithm, *args, **kwargs):
        """Generate a batch of data."""
        data_list = []

        for _ in range(num_samples):
            data = self._sample_data(*args, **kwargs)
            clrs_data = algorithm(*data)
            data.append(clrs_data)

        # Batch and pad trajectories to max(T).
        data = collate(data_list)
        return data

#   def next(self, batch_size: Optional[int] = None) -> Feedback:
#     """Subsamples trajectories from the pre-generated dataset.

#     Args:
#       batch_size: Optional batch size. If `None`, returns entire dataset.

#     Returns:
#       Subsampled trajectories.
#     """
#     if batch_size:
#       if self._num_samples < 0:  # generate on the fly
#         min_length = self.max_steps if self._track_max_steps else 0
#         inputs, outputs, hints, lengths = self._make_batch(
#             batch_size,
#             self._spec,
#             min_length,
#             self._algorithm,
#             *self._args,
#             **self._kwargs,
#         )
#         if self._track_max_steps and hints[0].data.shape[0] > self.max_steps:
#           logging.warning('Increasing hint lengh from %i to %i',
#                           self.max_steps, hints[0].data.shape[0])
#           self.max_steps = hints[0].data.shape[0]
#       else:
#         if batch_size > self._num_samples:
#           raise ValueError(
#               f'Batch size {batch_size} > dataset size {self._num_samples}.')

#         # Returns a fixed-size random batch.
#         indices = self._rng.choice(self._num_samples, (batch_size,),
#                                    replace=True)
#         inputs = _subsample_data(self._inputs, indices, axis=0)
#         outputs = _subsample_data(self._outputs, indices, axis=0)
#         hints = _subsample_data(self._hints, indices, axis=1)
#         lengths = self._lengths[indices]

#     else:
#       # Returns the full dataset.
#       assert self._num_samples >= 0
#       inputs = self._inputs
#       hints = self._hints
#       lengths = self._lengths
#       outputs = self._outputs

#     return Feedback(Features(inputs, hints, lengths), outputs)

#   @abc.abstractmethod
#   def _sample_data(self, length: int, *args, **kwargs) -> List[_Array]:
#     pass

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
#     if not directed:
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

# class ScheduleSampler(Sampler):
#   """Sorting sampler. Generates a random sequence of U[0, 1]."""

#   def _sample_data(
#       self,
#       length: int,
#       low: float = 0.,
#       high: float = 1.,
#   ):
#     arr = self._random_sequence(length=length, low=low, high=high)
#     return [arr]