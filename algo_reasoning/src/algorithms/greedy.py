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

"""Greedy algorithm generators.

Currently implements the following:
- Activity selection (Gavril, 1972)
- Task scheduling (Lawler, 1985)

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


def activity_selector(s, f, nb_nodes, *args, **kwargs):
    """Activity selection (Gavril, 1972)."""
    data = AlgorithmicData(algorithm="activity_selector", *args, **kwargs)
    
    data.set_inputs({
        's': s.clone(),
        'f': f.clone(),
    }, nb_nodes, inplace=True)

    A_pos = torch.arange(s.size(0))
    A = torch.zeros(s.shape[0])

    data.increase_hints({
        'pred_h': probe_array(A_pos.clone()),
        'selected_h': A.clone(),
        'm': mask_one(0, A_pos.size(0)),
        'k': mask_one(0, A_pos.size(0))
    }, inplace=True)
    
    ind = torch.argsort(f)
    A[ind[0]] = 1
    k = ind[0].item()

    data.increase_hints({
        'pred_h': probe_array(A_pos.clone()),
        'selected_h': A.clone(),
        'm': mask_one(ind[0], A_pos.size(0)),
        'k': mask_one(k, A_pos.size(0))
    }, inplace=True)

    for m in range(1, s.shape[0]):
        if s[ind[m]].item() >= f[k].item():
            A[ind[m]] = 1
            k = ind[m].item()
        
        data.increase_hints({
            'pred_h': probe_array(A_pos.clone()),
            'selected_h': A.clone(),
            'm': mask_one(ind[m], A_pos.size(0)),
            'k': mask_one(k, A_pos.size(0))
        }, inplace=True)

    data.set_outputs({
        'selected': A.clone(),
    }, inplace=True)

    return data

def task_scheduling(d, w, nb_nodes, *args, **kwargs):
    """Task scheduling (Lawler, 1985)."""
    data = AlgorithmicData(algorithm="task_scheduling", *args, **kwargs)
    
    data.set_inputs({
        'd': d.clone(),
        'w': w.clone(),
    }, nb_nodes, inplace=True)

    A_pos = torch.arange(d.shape[0])
    A = torch.zeros(d.shape[0])
    
    data.increase_hints({
        'pred_h': probe_array(A_pos.clone()),
        'selected_h': A.clone(),
        'i': mask_one(0, A_pos.size(0)),
        't': torch.tensor(0)
    }, inplace=True)

    ind = torch.argsort(-w)
    A[ind[0]] = 1
    t = 1

    data.increase_hints({
        'pred_h': probe_array(A_pos.clone()),
        'selected_h': A.clone(),
        'i': mask_one(ind[0], A_pos.size(0)),
        't': torch.tensor(t)
    }, inplace=True)

    for i in range(1, d.shape[0]):
        if t < d[ind[i]].item():
            A[ind[i]] = 1
            t += 1

        data.increase_hints({
            'pred_h': probe_array(A_pos.clone()),
            'selected_h': A.clone(),
            'i': mask_one(ind[i], A_pos.size(0)),
            't': torch.tensor(t)
        }, inplace=True)
    
    data.set_outputs({
        'selected': A.clone(),
    }, inplace=True)

    return data