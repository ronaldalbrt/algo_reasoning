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

"""Divide and conquer algorithm generators.

Currently implements the following:
- Maximum subarray
- Kadane's variant of Maximum subarray (Bentley, 1984)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# For the original implementation of such algorithms, please refer to: https://github.com/google-deepmind/clrs

# This implementation is pretty much the same as the original one, 
# but with some minor changes in the way the data is handled to better fit the Dataset in torch framework format.
# As the original implementation is made in the Haiku framework, which is a JAX-based framework,

# Modifications were made by Ronald Albert (https://www.pesc.coppe.ufrj.br/index.php/pt-BR/pessoas/details/18/2955), throughout the course of his Master's degree in Computer Science 
# at the Federal University of Rio de Janeiro (UFRJ), Brazil.

import torch

from algo_reasoning.src.data import AlgorithmicData
from algo_reasoning.src.probing import probe_array, mask_one


def find_maximum_subarray(A, nb_nodes, *args, **kwargs):
    """Maximum subarray."""
    data = AlgorithmicData(algorithm="find_maximum_subarray", *args, **kwargs)

    data.set_inputs({
        'key': A.clone()
    }, nb_nodes, inplace=True)

    A_pos = torch.arange(A.size(0))
    low = 0
    high = A.size(0) - 1

    def recursive_find_maximum_subarray(A, A_pos, low, high):

        def find_max_crossing_subarray(A, A_pos, low, mid, high, left_ctx, right_ctx):
            (left_low, left_high, l_ctx_sum) = left_ctx
            (right_low, right_high, r_ctx_sum) = right_ctx
            left_sum = A[mid].item() - 0.1
            sum_ = 0

            data.increase_hints({
                'pred_h': probe_array(A_pos.clone()),
                'low': mask_one(low, A.size(0)),
                'high': mask_one(high, A.size(0)),
                'mid': mask_one(mid, A.size(0)),
                'left_low': mask_one(left_low, A.size(0)),
                'left_high': mask_one(left_high, A.size(0)),
                'left_sum': torch.tensor(l_ctx_sum),
                'right_low': mask_one(right_low, A.size(0)),
                'right_high': mask_one(right_high, A.size(0)),
                'right_sum': torch.tensor(r_ctx_sum),
                'cross_low': mask_one(mid, A.size(0)),
                'cross_high': mask_one(mid + 1, A.size(0)),
                'cross_sum': A[mid] + A[mid + 1] - 0.2,
                'ret_low': mask_one(low, A.size(0)),
                'ret_high': mask_one(high, A.size(0)),
                'ret_sum': torch.tensor(0.0),
                'i': mask_one(mid, A.size(0)),
                'j': mask_one(mid + 1, A.size(0)),
                'sum': torch.tensor(0.0),
                'left_x_sum': A[mid] - 0.1,
                'right_x_sum': A[mid + 1] - 0.1,
                'phase': mask_one(1, 3)
            }, inplace=True)

            for i in range(mid, low - 1, -1):
                sum_ += A[i].item()
                if sum_ > left_sum:
                    left_sum = sum_
                    max_left = i
                
                data.increase_hints({
                    'pred_h': probe_array(A_pos.clone()),
                    'low': mask_one(low, A.size(0)),
                    'high': mask_one(high, A.size(0)),
                    'mid': mask_one(mid, A.size(0)),
                    'left_low': mask_one(left_low, A.size(0)),
                    'left_high': mask_one(left_high, A.size(0)),
                    'left_sum': torch.tensor(l_ctx_sum),
                    'right_low': mask_one(right_low, A.size(0)),
                    'right_high': mask_one(right_high, A.size(0)),
                    'right_sum': torch.tensor(r_ctx_sum),
                    'cross_low': mask_one(max_left, A.size(0)),
                    'cross_high': mask_one(mid + 1, A.size(0)),
                    'cross_sum': left_sum + A[mid + 1] - 0.1,
                    'ret_low': mask_one(low, A.size(0)),
                    'ret_high': mask_one(high, A.size(0)),
                    'ret_sum': torch.tensor(0.0),
                    'i': mask_one(i, A.size(0)),
                    'j': mask_one(mid + 1, A.size(0)),
                    'sum': torch.tensor(sum_),
                    'left_x_sum': torch.tensor(left_sum),
                    'right_x_sum': A[mid + 1] - 0.1,
                    'phase': mask_one(1, 3)
                }, inplace=True)

        
                right_sum = A[mid + 1].item() - 0.1
                sum_ = 0

                data.increase_hints({
                    'pred_h': probe_array(A_pos.clone()),
                    'low': mask_one(low, A.size(0)),
                    'high': mask_one(high, A.size(0)),
                    'mid': mask_one(mid, A.size(0)),
                    'left_low': mask_one(left_low, A.size(0)),
                    'left_high': mask_one(left_high, A.size(0)),
                    'left_sum': left_sum,
                    'right_low': mask_one(right_low, A.size(0)),
                    'right_high': mask_one(right_high, A.size(0)),
                    'right_sum': right_sum,
                    'cross_low': mask_one(max_left, A.size(0)),
                    'cross_high': mask_one(mid + 1, A.size(0)),
                    'cross_sum': left_sum + right_sum,
                    'ret_low': mask_one(low, A.size(0)),
                    'ret_high': mask_one(high, A.size(0)),
                    'ret_sum': torch.tensor(0.0),
                    'i': mask_one(i, A.size(0)),
                    'j': mask_one(mid + 1, A.size(0)),
                    'sum': torch.tensor(0.0),
                    'left_x_sum': left_sum,
                    'right_x_sum': A[mid + 1] - 0.1,
                    'phase': mask_one(2, 3)
                }, inplace=True)

            for j in range(mid + 1, high + 1):
                sum_ += A[j].item()
                if sum_ > right_sum:
                    right_sum = sum_
                    max_right = j

                data.increase_hints({
                    'pred_h': probe_array(A_pos.clone()),
                    'low': mask_one(low, A.size(0)),
                    'high': mask_one(high, A.size(0)),
                    'mid': mask_one(mid, A.size(0)),
                    'left_low': mask_one(left_low, A.size(0)),
                    'left_high': mask_one(left_high, A.size(0)),
                    'left_sum': torch.tensor(left_sum),
                    'right_low': mask_one(right_low, A.size(0)),
                    'right_high': mask_one(right_high, A.size(0)),
                    'right_sum': torch.tensor(right_sum),
                    'cross_low': mask_one(max_left, A.size(0)),
                    'cross_high': mask_one(max_right, A.size(0)),
                    'cross_sum': torch.tensor(left_sum + right_sum),
                    'ret_low': mask_one(low, A.size(0)),
                    'ret_high': mask_one(high, A.size(0)),
                    'ret_sum': torch.tensor(0.0),
                    'i': mask_one(i, A.size(0)),
                    'j': mask_one(j, A.size(0)),
                    'sum': torch.tensor(sum_),
                    'left_x_sum': torch.tensor(left_sum),
                    'right_x_sum': torch.tensor(right_sum),
                    'phase': mask_one(2, 3)
                })
            
            return (max_left, max_right, left_sum + right_sum), (sum_, left_sum, right_sum)

        mid = (low + high) // 2

        if high == low:
            if A.size(0) == 1:
                data.set_outputs({
                    'start': mask_one(low, A.size(0)),
                    'end': mask_one(high, A.size(0))
                }, inplace=True)

                return (low, high, A[low].item())
            else:
                data.increase_hints({
                    'pred_h': probe_array(A_pos.clone()),
                    'low': mask_one(low, A.shape[0]),
                    'high': mask_one(high, A.shape[0]),
                    'mid': mask_one(mid, A.shape[0]),
                    'left_low': mask_one(low, A.shape[0]),
                    'left_high': mask_one(high, A.shape[0]),
                    'left_sum': torch.tensor(0.0),
                    'right_low': mask_one(low, A.shape[0]),
                    'right_high': mask_one(high, A.shape[0]),
                    'right_sum': torch.tensor(0.0),
                    'cross_low': mask_one(low, A.shape[0]),
                    'cross_high': mask_one(high, A.shape[0]),
                    'cross_sum': torch.tensor(0.0),
                    'ret_low': mask_one(low, A.shape[0]),
                    'ret_high': mask_one(high, A.shape[0]),
                    'ret_sum': A[low].clone(),
                    'i': mask_one(low, A.shape[0]),
                    'j': mask_one(high, A.shape[0]),
                    'sum': torch.tensor(0.0),
                    'left_x_sum': A[low] - 0.1,
                    'right_x_sum': A[high] - 0.1,
                    'phase': mask_one(0, 3)
                }, inplace=True)

            return (low, high, A[low].item())
        else:
            data.increase_hints({
                    'pred_h': probe_array(A_pos.clone()),
                    'low': mask_one(low, A.shape[0]),
                    'high': mask_one(high, A.shape[0]),
                    'mid': mask_one(mid, A.shape[0]),
                    'left_low': mask_one(low, A.shape[0]),
                    'left_high': mask_one(mid, A.shape[0]),
                    'left_sum': torch.tensor(0.0),
                    'right_low': mask_one(mid + 1, A.shape[0]),
                    'right_high': mask_one(high, A.shape[0]),
                    'right_sum': torch.tensor(0.0),
                    'cross_low': mask_one(mid, A.shape[0]),
                    'cross_high': mask_one(mid + 1, A.shape[0]),
                    'cross_sum': A[mid] + A[mid + 1] - 0.2,
                    'ret_low': mask_one(low, A.shape[0]),
                    'ret_high': mask_one(high, A.shape[0]),
                    'ret_sum': torch.tensor(0.0),
                    'i': mask_one(mid, A.shape[0]),
                    'j': mask_one(mid + 1, A.shape[0]),
                    'sum': torch.tensor(0.0),
                    'left_x_sum': A[mid] - 0.1,
                    'right_x_sum': A[mid + 1] - 0.1,
                    'phase': mask_one(0, 3)
                })

            left_low, left_high, left_sum = recursive_find_maximum_subarray(A, A_pos, low, mid)

            data.increase_hints({
                'pred_h': probe_array(A_pos.clone()),
                'low': mask_one(low, A.shape[0]),
                'high': mask_one(high, A.shape[0]),
                'mid': mask_one(mid, A.shape[0]),
                'left_low': mask_one(left_low, A.shape[0]),
                'left_high': mask_one(left_high, A.shape[0]),
                'left_sum': torch.tensor(left_sum),
                'right_low': mask_one(mid + 1, A.shape[0]),
                'right_high': mask_one(high, A.shape[0]),
                'right_sum': torch.tensor(0.0),
                'cross_low': mask_one(mid, A.shape[0]),
                'cross_high': mask_one(mid + 1, A.shape[0]),
                'cross_sum': A[mid] + A[mid + 1] - 0.2,
                'ret_low': mask_one(low, A.shape[0]),
                'ret_high': mask_one(high, A.shape[0]),
                'ret_sum': torch.tensor(0.0),
                'i': mask_one(mid, A.shape[0]),
                'j': mask_one(mid + 1, A.shape[0]),
                'sum': torch.tensor(0.0),
                'left_x_sum': A[mid] - 0.1,
                'right_x_sum': A[mid + 1] - 0.1,
                'phase': mask_one(0, 3)
        })

            right_low, right_high, right_sum = recursive_find_maximum_subarray(A, A_pos, mid + 1, high)

            data.increase_hints({
                'pred_h': probe_array(A_pos.clone()),
                'low': mask_one(low, A.shape[0]),
                'high': mask_one(high, A.shape[0]),
                'mid': mask_one(mid, A.shape[0]),
                'left_low': mask_one(left_low, A.shape[0]),
                'left_high': mask_one(left_high, A.shape[0]),
                'left_sum': torch.tensor(left_sum),
                'right_low': mask_one(right_low, A.shape[0]),
                'right_high': mask_one(right_high, A.shape[0]),
                'right_sum': torch.tensor(right_sum),
                'cross_low': mask_one(mid, A.shape[0]),
                'cross_high': mask_one(mid + 1, A.shape[0]),
                'cross_sum': A[mid] + A[mid + 1] - 0.2,
                'ret_low': mask_one(low, A.shape[0]),
                'ret_high': mask_one(high, A.shape[0]),
                'ret_sum': torch.tensor(0.0),
                'i': mask_one(mid, A.shape[0]),
                'j': mask_one(mid + 1, A.shape[0]),
                'sum': torch.tensor(0.0),
                'left_x_sum': A[mid] - 0.1,
                'right_x_sum': A[mid + 1] - 0.1,
                'phase': mask_one(0, 3)
            })

            (cross_low, cross_high, cross_sum), (x_sum, x_left, x_right) = find_max_crossing_subarray(A, A_pos, low, mid, high, (left_low, left_high, left_sum), (right_low, right_high, right_sum))
            if left_sum >= right_sum and left_sum >= cross_sum:
                best = (left_low, left_high, left_sum)
            elif right_sum >= left_sum and right_sum >= cross_sum:
                best = (right_low, right_high, right_sum)
            else:
                best = (cross_low, cross_high, cross_sum)

            if low == 0 and high == A.shape[0] - 1:
                data.set_outputs({
                    'start': mask_one(best[0], A.size(0)),
                    'end': mask_one(best[1], A.size(0))
                }, inplace=True)

                return best
            else:
                data.increase_hints({
                    'pred_h': probe_array(A_pos.clone()),
                    'low': mask_one(low, A.shape[0]),
                    'high': mask_one(high, A.shape[0]),
                    'mid': mask_one(mid, A.shape[0]),
                    'left_low': mask_one(left_low, A.shape[0]),
                    'left_high': mask_one(left_high, A.shape[0]),
                    'left_sum': torch.tensor(left_sum),
                    'right_low': mask_one(right_low, A.shape[0]),
                    'right_high': mask_one(right_high, A.shape[0]),
                    'right_sum': torch.tensor(right_sum),
                    'cross_low': mask_one(cross_low, A.shape[0]),
                    'cross_high': mask_one(cross_high, A.shape[0]),
                    'cross_sum': torch.tensor(cross_sum),
                    'ret_low': mask_one(best[0], A.shape[0]),
                    'ret_high': mask_one(best[1], A.shape[0]),
                    'ret_sum': torch.tensor(best[2]),
                    'i': mask_one(low, A.shape[0]),
                    'j': mask_one(high, A.shape[0]),
                    'sum': torch.tensor(x_sum),
                    'left_x_sum': torch.tensor(x_left),
                    'right_x_sum': torch.tensor(x_right),
                    'phase': mask_one(0, 3)
                })

                return best
    
    recursive_find_maximum_subarray(A, A_pos, low, high)

    return data


def find_maximum_subarray_kadane(A, nb_nodes, *args, **kwargs):
    """Kadane's variant of Maximum subarray (Bentley, 1984)."""

    data = AlgorithmicData(algorithm="find_maximum_subarray_kadane", *args, **kwargs)

    data.set_inputs({
        'key': A.clone()
    }, nb_nodes, inplace=True)

    A_pos = torch.arange(A.size(0))
    
    data.increase_hints({
        'pred_h': probe_array(A_pos.clone()),
        'best_low': mask_one(0, A.shape[0]),
        'best_high': mask_one(0, A.shape[0]),
        'best_sum': A[0].clone(),
        'i': mask_one(0, A.shape[0]),
        'j': mask_one(0, A.shape[0]),
        'sum': A[0].clone()
    }, inplace=True)

    best_low = 0
    best_high = 0
    best_sum = A[0].item()
    i = 0
    sum_ = A[0].item()

    for j in range(1, A.shape[0]):
        x = A[j].item()
        if sum_ + x >= x:
            sum_ += x
        else:
            i = j
            sum_ = x
        if sum_ > best_sum:
            best_low = i
            best_high = j
            best_sum = sum_

        data.increase_hints({
            'pred_h': probe_array(A_pos.clone()),
            'best_low': mask_one(best_low, A.shape[0]),
            'best_high': mask_one(best_high, A.shape[0]),
            'best_sum': torch.tensor(best_sum),
            'i': mask_one(i, A.shape[0]),
            'j': mask_one(j, A.shape[0]),
            'sum': torch.tensor(sum_)
        }, inplace=True)

    data.set_outputs({
        'start': mask_one(best_low, A.size(0)),
        'end': mask_one(best_high, A.size(0))
    }, inplace=True)
  
    return data
