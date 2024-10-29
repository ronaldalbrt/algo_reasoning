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

"""Searching algorithm generators.

Currently implements the following:
- Minimum
- Binary search
- Quickselect (Hoare, 1961)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# pylint: disable=invalid-name
import torch

from algo_reasoning.src.data import AlgorithmicData
from algo_reasoning.src.probing import probe_array, mask_one

def minimum(A, nb_nodes, *args, **kwargs):
    """Minimum."""
    data = AlgorithmicData(algorithm="minimum", *args, **kwargs)

    A_pos = torch.arange(A.size(0))

    data.set_inputs({
        'key': A.clone()
    }, nb_nodes, inplace=True)
    
    data.increase_hints({
        'pred_h': probe_array(A_pos.clone()),
        'min_h': mask_one(0, A.size(0)),
        'i': mask_one(0, A.size(0))
    }, inplace=True)

    min_ = 0
    for i in range(1, A.shape[0]):
        if A[min_] > A[i]:
            min_ = i

        data.increase_hints({
            'pred_h': probe_array(A_pos.clone()),
            'min_h': mask_one(min_, A.size(0)),
            'i': mask_one(i, A.size(0))
        }, inplace=True)

    data.set_outputs({
        'min': mask_one(min_, A.size(0))
    }, inplace=True)

    return data


def binary_search(x, A, nb_nodes, *args, **kwargs):
    """Binary search."""
    data = AlgorithmicData(algorithm="binary_search", *args, **kwargs)

    T_pos = torch.arange(A.size(0))

    data.set_inputs({
        'key': A.clone(),
        'target': torch.tensor(x)
    }, nb_nodes, inplace=True)

    data.increase_hints({
        'pred_h': probe_array(T_pos.clone()),
        'low': mask_one(0, A.size(0)),
        'high': mask_one(A.size(0) - 1, A.size(0)),
        'mid': mask_one((A.size(0) - 1) // 2, A.size(0))
    }, inplace=True)

    low = 0
    high = A.size(0) - 1  
    while low < high:
        mid = (low + high) // 2
        if x <= A[mid]:
            high = mid
        else:
            low = mid + 1

        data.increase_hints({
            'pred_h': probe_array(T_pos.clone()),
            'low': mask_one(low, A.size(0)),
            'high': mask_one(high, A.size(0)),
            'mid': mask_one((low + high) // 2, A.size(0))
        }, inplace=True)

    data.set_outputs({
        'return': mask_one(high, A.size(0))
    }, inplace=True)

    return data

def quickselect(A, nb_nodes, *args, **kwargs):
    """Quickselect (Hoare, 1961)."""
    
    data = AlgorithmicData(algorithm="quickselect", *args, **kwargs)

    data.set_inputs({
        'key': A.clone()
    }, nb_nodes, inplace=True)

    A_pos = torch.arange(A.size(0))
    p = 0
    r = A.size(0) - 1
    i = A.size(0) // 2

    def recursive_quickselect(A, A_pos, p, r, i):
        def partition(A, A_pos, p, r, target):
            x = A[r].item()
            i = p - 1
            for j in range(p, r):
                if A[j] <= x:
                    i += 1
                    A[i], A[j] = A[j].item(), A[i].item()
                    A_pos[i], A_pos[j] = A_pos[j].item(), A_pos[i].item()
                
                data.increase_hints({
                    'pred_h': probe_array(A_pos.clone()),
                    'p': mask_one(A_pos[p].item(), A.size(0)),
                    'r': mask_one(A_pos[r].item(), A.size(0)),
                    'i': mask_one(A_pos[i + 1].item(), A.size(0)),
                    'j': mask_one(A_pos[j].item(), A.size(0)),
                    'i_rank': torch.tensor((i + 1) * 1.0 / A.size(0)),
                    'target': torch.tensor(target * 1.0 / A.size(0)),
                    'pivot': mask_one(A_pos[r].item(), A.size(0)),
                }, inplace=True)

            A[i + 1], A[r] = A[r].item(), A[i + 1].item()
            A_pos[i + 1], A_pos[r] = A_pos[r].item(), A_pos[i + 1].item()

            data.increase_hints({
                'pred_h': probe_array(A_pos.clone()),
                'p': mask_one(A_pos[p].item(), A.size(0)),
                'r': mask_one(A_pos[r].item(), A.size(0)),
                'i': mask_one(A_pos[i + 1].item(), A.size(0)),
                'j': mask_one(A_pos[r].item(), A.size(0)),
                'i_rank': torch.tensor((i + 1 - p) * 1.0 / A.size(0)),
                'target': torch.tensor(target * 1.0 / A.size(0)),
                'pivot': mask_one(A_pos[i + 1].item(), A.size(0)),
            }, inplace=True)

            return i + 1

        q = partition(A, A_pos, p, r, i)
        k = q - p
        
        if i == k:
            data.set_outputs({
                'median': mask_one(A_pos[q].item(), A.size(0))
            }, inplace=True)

        elif i < k:
            recursive_quickselect(A, A_pos, p, q - 1, i)
        else:
            recursive_quickselect(A, A_pos, q + 1, r, i - k - 1)

    recursive_quickselect(A, A_pos, p, r, i)
  

    return data
