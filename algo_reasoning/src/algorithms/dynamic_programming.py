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

"""Dynamic programming algorithm generators.

Currently implements the following:
- Matrix-chain multiplication
- Longest common subsequence
- Optimal binary search tree (Aho et al., 1974)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# pylint: disable=invalid-name
import torch

from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.probing import probe_array, strings_id, array_cat, strings_pair, strings_pair_cat, strings_pred


def matrix_chain_order(p, nb_nodes, *args, **kwargs):
    """Matrix-chain multiplication."""

    data = CLRSData(algorithm="matrix_chain_order", *args, **kwargs)

    A_pos = torch.arange(p.size(0))

    data.set_inputs({
        'p': p.clone()
    }, nb_nodes, inplace=True)

    m = torch.zeros((p.size(0), p.size(0)))
    s = torch.zeros((p.size(0), p.size(0)))
    msk = torch.zeros((p.size(0), p.size(0)))
    for i in range(1, p.size(0)):
        m[i, i] = 0
        msk[i, i] = 1
    while True:
        prev_m = m.clone()
        prev_msk = msk.clone()

        data.increase_hints({
           'pred_h': probe_array(A_pos.clone()),
            'm': prev_m.clone(),
            's_h': s.clone(),
            'msk': msk.clone()
        }, inplace=True)

        for i in range(1, p.size(0)):
            for j in range(i + 1, p.size(0)):
                flag = prev_msk[i, j].item()
                for k in range(i, j):
                    if prev_msk[i, k] == 1 and prev_msk[k + 1, j] == 1:
                        msk[i, j] = 1
                        q = (prev_m[i, k] + prev_m[k + 1, j] + p[i - 1] * p[k] * p[j]).item()
                        if flag == 0 or q < m[i, j].item():
                            m[i, j] = q
                            s[i, j] = k
                            flag = 1
        if torch.all(prev_m == m):
            break
    
    data.set_outputs({
        's': s.clone()
    }, inplace=True)

    return data


def lcs_length(x, y, nb_nodes, *args, **kwargs):
    """Longest common subsequence."""
    data = CLRSData(algorithm="lcs_length", *args, **kwargs)

    x_pos = torch.arange(x.size(0))
    y_pos = torch.arange(y.size(0))
    b = torch.zeros((x.size(0), y.size(0)))
    c = torch.zeros((x.size(0), y.size(0)))

    data.set_inputs({
        'string': strings_id(x_pos, y_pos),
        'key': array_cat(torch.concatenate((x.clone(), y.clone())), 4)
    }, nb_nodes, inplace=True)

    for i in range(x.size(0)):
        if x[i].item() == y[0].item():
            c[i, 0] = 1
            b[i, 0] = 0
        elif i > 0 and c[i - 1, 0].item() == 1:
            c[i, 0] = 1
            b[i, 0] = 1
        else:
            c[i, 0] = 0
            b[i, 0] = 1
    for j in range(y.size(0)):
        if x[0].item() == y[j].item():
            c[0, j] = 1
            b[0, j] = 0
        elif j > 0 and c[0, j - 1].item() == 1:
            c[0, j] = 1
            b[0, j] = 2
        else:
            c[0, j] = 0
            b[0, j] = 1

    while True:
        prev_c = c.clone()

        data.increase_hints({
                'pred_h': strings_pred(x_pos, y_pos),
                'b_h': strings_pair_cat(b.clone(), 3),
                'c': strings_pair(prev_c)
        }, inplace=True)

        for i in range(1, x.shape[0]):
            for j in range(1, y.shape[0]):
                if x[i].item() == y[j].item():
                    c[i, j] = prev_c[i - 1, j - 1].item() + 1
                    b[i, j] = 0
                elif prev_c[i - 1, j].item() >= prev_c[i, j - 1].item():
                    c[i, j] = prev_c[i - 1, j].item()
                    b[i, j] = 1
                else:
                    c[i, j] = prev_c[i, j - 1].item()
                    b[i, j] = 2
        if torch.all(prev_c == c):
            break
    
    data.set_outputs({
        'b': strings_pair_cat(b.clone(), 3)
    }, inplace=True)

    return data


def optimal_bst(p, q, nb_nodes, *args, **kwargs):
    """Optimal binary search tree (Aho et al., 1974)."""

    data = CLRSData(algorithm="optimal_bst", *args, **kwargs)

    A_pos = torch.arange(q.shape[0])
    p_cpy = torch.zeros(q.shape[0])
    p_cpy[:-1] = p.clone()

    data.set_inputs({
        'p': p_cpy.clone(),
        'q': q.clone()
    }, nb_nodes, inplace=True)

    e = torch.zeros((q.size(0), q.size(0)))
    w = torch.zeros((q.size(0), q.size(0)))
    root = torch.zeros((q.size(0), q.size(0)))
    msks = torch.zeros((q.size(0), q.size(0)))

    for i in range(q.size(0)):
        e[i, i] = q[i].item()
        w[i, i] = q[i].item()
        msks[i, i] = 1

    data.increase_hints({
        'pred_h': probe_array(A_pos.clone()),
        'root_h': root.clone(),
        'e': e.clone(),
        'w': w.clone(),
        'msk': msks.clone()
    }, inplace=True)

    for l in range(1, p.size(0) + 1):
        for i in range(p.size(0) - l + 1):
            j = i + l
            e[i, j] = 1e9
            w[i, j] = w[i, j - 1].item() + p[j - 1].item() + q[j].item()
            for r in range(i, j):
                t = e[i, r].item() + e[r + 1, j].item() + w[i, j].item()
                if t < e[i, j].item():
                    e[i, j] = t
                    root[i, j] = r
            msks[i, j] = 1

        data.increase_hints({
            'pred_h': probe_array(A_pos.clone()),
            'root_h': root.clone(),
            'e': e.clone(),
            'w': w.clone(),
            'msk': msks.clone()
        }, inplace=True)

    data.set_outputs({
        'root': root.clone()
    }, inplace=True)

    return data
