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

"""Sorting algorithm generators.

Currently implements the following:
- Insertion sort
- Bubble sort
- Heapsort (Williams, 1964)
- Quicksort (Hoare, 1962)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# pylint: disable=invalid-name
import torch

from algo_reasoning.src.data import CLRSData

# TODO: Create probing file
def probe_array(A_pos):
    """Constructs an `array` probe."""
    probe = torch.arange(A_pos.size(0))
    for i in range(1, A_pos.size(0)):
        probe[A_pos[i]] = A_pos[i - 1]

    return probe

def mask_one(i, n):
    """Constructs a `mask_one` probe."""
    assert n > i
    probe = torch.zeros(n)
    probe[i] = 1
    return probe


def heap(A_pos, heap_size):
    """Constructs a `heap` probe."""
    assert heap_size > 0
    
    probe = torch.arange(A_pos.size(0))
    
    for i in range(1, heap_size):
        probe[A_pos[i]] = A_pos[(i - 1) // 2]
    
    return probe

def insertion_sort(A, nb_nodes, *args, **kwargs):
    """Insertion sort."""
    data = CLRSData(algorithm="insertion_sort", *args, **kwargs)
    
    data = data.set_inputs({
        'key': A
    }, nb_nodes)

    A_pos = torch.arange(A.size(0))

    data = data.increase_hints({
        'pred_h': probe_array(A_pos.clone()),
        'i': mask_one(0, A.size(0)),
        'j': mask_one(0, A.size(0))
    })

    for j in range(1, A.size(0)):
        key = A[j].item()
        # Insert A[j] into the sorted sequence A[1 .. j - 1]
        i = j - 1
        while i >= 0 and A[i] > key:
            A[i + 1] = A[i].item()
            A_pos[i + 1] = A_pos[i].item()
            i -= 1
        A[i + 1] = key
        stor_pos = A_pos[i + 1].item()
        A_pos[i + 1] = j

        data = data.increase_hints({
            'pred_h': probe_array(A_pos.clone()),
            'i': mask_one(stor_pos, A.size(0)),
            'j': mask_one(j, A.size(0))
        })
    
    data = data.set_outputs({
        'pred': probe_array(A_pos.clone()),
    })

    return data


def bubble_sort(A, nb_nodes, *args, **kwargs):
    """Bubble sort."""
    data = CLRSData(algorithm="insertion_sort", *args, **kwargs)
    
    data = data.set_inputs({
        'key': A
    }, nb_nodes)

    A_pos = torch.arange(A.size(0))

    data = data.increase_hints({
        'pred_h': probe_array(A_pos.clone()),
        'i': mask_one(0, A.size(0)),
        'j': mask_one(0, A.size(0))
    })

    for i in range(A.size(0) - 1):
        for j in reversed(range(i + 1, A.size(0))):
            if A[j] < A[j - 1]:
                tmp = A[j].item()
                A[j] = A[j - 1].item()
                A[j - 1] = tmp

                tmp = A_pos[j].item()
                A_pos[j] = A_pos[j - 1].item()
                A_pos[j - 1] = tmp

                data = data.increase_hints({
                    'pred_h': probe_array(A_pos.clone()),
                    'i': mask_one(A_pos[i], A.size(0)),
                    'j': mask_one(A_pos[j], A.size(0))
                })

    data = data.set_outputs({
       'pred': probe_array(A_pos.clone())
    })

    return data


def heapsort(A, nb_nodes, *args, **kwargs):
    """Heapsort (Williams, 1964)."""
    data = CLRSData(algorithm="heapsort", *args, **kwargs)
    
    data.set_inputs({
        'key': A
    }, nb_nodes, inplace=True)

    A_pos = torch.arange(A.size(0))

    data.increase_hints({
          'pred_h': probe_array(A_pos.clone()),
          'parent': heap(A_pos.clone(), A.size(0)),
          'i': mask_one(A.size(0) - 1, A.size(0)),
          'j': mask_one(A.size(0) - 1, A.size(0)),
          'largest': mask_one(A.size(0) - 1, A.size(0)),
          'heap_size': mask_one(A.size(0) - 1, A.size(0)),
          'phase': mask_one(0, 3)
    }, inplace=True)


    def max_heapify(A, i, heap_size, ind, phase):
        l = 2 * i + 1
        r = 2 * i + 2
        if l < heap_size and A[l] > A[i]:
            largest = l
        else:
            largest = i
        if r < heap_size and A[r] > A[largest]:
            largest = r
        if largest != i:
            A[i], A[largest] = A[largest].item(), A[i].item()
            A_pos[i], A_pos[largest] = A_pos[largest].item(), A_pos[i].item()

        data.increase_hints({
          'pred_h': probe_array(A_pos.clone()),
          'parent': heap(A_pos.clone(), heap_size),
          'i': mask_one(A_pos[ind].item(), A.size(0)),
          'j': mask_one(A_pos[i].item(), A.size(0)),
          'largest': mask_one(A_pos[largest].item(), A.size(0)),
          'heap_size': mask_one(A_pos[heap_size - 1].item(), A.size(0)),
          'phase': mask_one(phase, 3)
        }, inplace=True)

        if largest != i:
            max_heapify(A, largest, heap_size, ind, phase)

    def build_max_heap(A):
        for i in reversed(range(A.size(0))):
            max_heapify(A, i, A.size(0), i, 0)

    build_max_heap(A)
    heap_size = A.size(0)

    for i in reversed(range(1, A.size(0))):
        A[0], A[i] = A[i].item(), A[0].item()
        A_pos[0], A_pos[i] = A_pos[i].item(), A_pos[0].item()

        heap_size -= 1

        data.increase_hints({
          'pred_h': probe_array(A_pos.clone()),
          'parent': heap(A_pos.clone(), heap_size),
          'i': mask_one(A_pos[0].item(), A.size(0)),
          'j': mask_one(A_pos[i].item(), A.size(0)),
          'largest': mask_one(0, A.size(0)),
          'heap_size': mask_one(A_pos[heap_size - 1].item(), A.size(0)),
          'phase': mask_one(1, 3)
        }, inplace=True)


        max_heapify(A, 0, heap_size, i, 2) 

    data.set_outputs({
       'pred': probe_array(A_pos.clone())
    }, inplace=True)

    return data

def quicksort(A, nb_nodes, *args, **kwargs):
    """Quicksort (Hoare, 1962)."""
    
    data = CLRSData(algorithm="quicksort", *args, **kwargs)

    data.set_inputs({
        'key': A
    }, nb_nodes, inplace=True)

    A_pos = torch.arange(A.size(0))
    p = 0
    r = A.size(0) - 1

    def recursive_quicksort(A, A_pos, p, r):
        def partition(A, A_pos, p, r):
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
                    'j': mask_one(A_pos[j].item(), A.size(0))
                }, inplace=True)

            A[i + 1], A[r] = A[r].item(), A[i + 1].item()
            A_pos[i + 1], A_pos[r] = A_pos[r].item(), A_pos[i + 1].item()

            data.increase_hints({
                'pred_h': probe_array(A_pos.clone()),
                'p': mask_one(A_pos[p].item(), A.size(0)),
                'r': mask_one(A_pos[r].item(), A.size(0)),
                'i': mask_one(A_pos[i + 1].item(), A.size(0)),
                'j': mask_one(A_pos[r].item(), A.size(0))
            }, inplace=True)

            return i + 1

        if p < r:
            q = partition(A, A_pos, p, r)
            recursive_quicksort(A, A_pos, p, q - 1)
            recursive_quicksort(A, A_pos, q + 1, r)

        if p == 0 and r == len(A) - 1:
            data.set_outputs({
                'pred': probe_array(A_pos.clone())
            }, inplace=True)

    recursive_quicksort(A, A_pos, p, r)

    return data