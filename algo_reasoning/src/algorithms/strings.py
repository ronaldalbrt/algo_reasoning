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

"""Strings algorithm generators.

Currently implements the following:
- Naive string matching
- Knuth-Morris-Pratt string matching (Knuth et al., 1977)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""

# For the original implementation of such algorithms, please refer to: https://github.com/google-deepmind/clrs

# This implementation is pretty much the same as the original one, 
# but with some minor changes in the way the data is handled to better fit the dataset in torch framework format.
# As the original implementation is made in the Haiku framework, which is a JAX-based framework,

# Modifications were made by Ronald Albert (https://www.pesc.coppe.ufrj.br/index.php/pt-BR/pessoas/details/18/2955), throughout the course of his Master's degree in Computer Science 
# at the Federal University of Rio de Janeiro (UFRJ), Brazil.

import torch

from algo_reasoning.src.data import AlgorithmicData
from algo_reasoning.src.probing import mask_one, array_cat, strings_id, strings_pred, strings_pi

_ALPHABET_SIZE = 4


def naive_string_matcher(T, P, nb_nodes, *args, **kwargs):
    """Naive string matching."""

    data = AlgorithmicData(algorithm='naive_string_matcher',  *args, **kwargs)

    T_pos = torch.arange(T.size(0))
    P_pos = torch.arange(P.size(0))

    _strings_id = strings_id(T_pos, P_pos)

    data.set_inputs({
          'string': _strings_id,
          'key': array_cat(torch.concatenate([T.clone(), P.clone()]), _ALPHABET_SIZE),
      }, nb_nodes, _strings_id=_strings_id)
    

    s = 0
    while s <= T.size(0) - P.size(0):
        i = s
        j = 0

        data.increase_hints({
                'pred_h': strings_pred(T_pos, P_pos),
                's': mask_one(s, T.size(0) + P.size(0)),
                'i': mask_one(i, T.size(0) + P.size(0)),
                'j': mask_one(T.size(0) + j, T.size(0) + P.size(0))
        })

        while True:
            if T[i] != P[j]:
                break
            elif j == P.size(0) - 1:
                data.set_outputs({
                    'match': mask_one(s, T.size(0) + P.size(0))
                })
                
                return data
            else:
                i += 1
                j += 1

                data.increase_hints({
                        'pred_h': strings_pred(T_pos, P_pos),
                        's': mask_one(s, T.size(0) + P.size(0)),
                        'i': mask_one(i, T.size(0) + P.size(0)),
                        'j': mask_one(T.size(0) + j, T.size(0) + P.size(0))
                    })

        s += 1


    data.set_outputs({
                    'match': mask_one(T.size(0), T.size(0) + P.size(0))
                })

    return data


def kmp_matcher(T, P, nb_nodes, *args, **kwargs):
    """Knuth-Morris-Pratt string matching (Knuth et al., 1977)."""

    data = AlgorithmicData(algorithm='kmp_matcher',  *args, **kwargs)

    T_pos = torch.arange(T.size(0))
    P_pos = torch.arange(P.size(0))

    _strings_id = strings_id(T_pos, P_pos)

    data.set_inputs({
          'string': _strings_id,
          'key': array_cat(torch.concatenate([T.clone(), P.clone()]), _ALPHABET_SIZE),
      }, nb_nodes, _strings_id=_strings_id)

    pi = torch.arange(P.size(0))
    is_reset = torch.zeros(P.size(0))

    k = 0
    k_reset = 1
    is_reset[0] = 1

    delta = 1 if P.size(0) > 1 else 0

    data.increase_hints({
        'pred_h': strings_pred(T_pos, P_pos),
        'pi': strings_pi(T_pos, P_pos, pi),
        'is_reset': torch.concatenate([torch.zeros(T.size(0)), is_reset.clone()]),
        'k': mask_one(T.size(0), T.size(0) + P.size(0)),
        'k_reset': torch.tensor(k_reset),
        'q': mask_one(T.size(0) + delta, T.size(0) + P.size(0)),
        'q_reset': torch.tensor(1),
        's': mask_one(0, T.size(0) + P.size(0)),
        'i': mask_one(0, T.size(0) + P.size(0)),
        'phase': torch.tensor(0)
    })

    for q in range(1, P.size(0)):
        while k_reset == 0 and P[k + 1].item() != P[q].item():
            if is_reset[k].item() == 1:
                k_reset = 1
                k = 0
            else:
                k = pi[k].item()

            data.increase_hints({
                'pred_h': strings_pred(T_pos, P_pos),
                'pi': strings_pi(T_pos, P_pos, pi),
                'is_reset': torch.concatenate([torch.zeros(T.size(0)), is_reset.clone()]),
                'k': mask_one(T.size(0) + k, T.size(0) + P.size(0)),
                'k_reset': torch.tensor(k_reset),
                'q': mask_one(T.size(0) + q, T.size(0) + P.size(0)),
                'q_reset': torch.tensor(1),
                's': mask_one(0, T.size(0) + P.size(0)),
                'i': mask_one(0, T.size(0) + P.size(0)),
                'phase': torch.tensor(0)
            })
        if k_reset == 1:
            k_reset = 0
            k = -1
        if P[k + 1].item() == P[q].item():
            k += 1
        if k == -1:
            k = 0
            k_reset = 1
            is_reset[q] = 1
        pi[q] = k

        data.increase_hints({
            'pred_h': strings_pred(T_pos, P_pos),
            'pi': strings_pi(T_pos, P_pos, pi),
            'is_reset': torch.concatenate([torch.zeros(T.size(0)), is_reset.clone()]),
            'k': mask_one(T.size(0) + k, T.size(0) + P.size(0)),
            'k_reset': torch.tensor(k_reset),
            'q': mask_one(T.size(0) + q, T.size(0) + P.size(0)),
            'q_reset': torch.tensor(1),
            's': mask_one(0, T.size(0) + P.size(0)),
            'i': mask_one(0, T.size(0) + P.size(0)),
            'phase': torch.tensor(0)
        })
    
    q = 0
    q_reset = 1
    s = 0
    for i in range(T.size(0)):
        if i >= P.size(0):
            s += 1

        data.increase_hints({
            'pred_h': strings_pred(T_pos, P_pos),
            'pi': strings_pi(T_pos, P_pos, pi),
            'is_reset': torch.concatenate([torch.zeros(T.size(0)), is_reset.clone()]),
            'k': mask_one(T.size(0) + k, T.size(0) + P.size(0)),
            'k_reset': torch.tensor(k_reset),
            'q': mask_one(T.size(0) + q, T.size(0) + P.size(0)),
            'q_reset': torch.tensor(q_reset),
            's': mask_one(s, T.size(0) + P.size(0)),
            'i': mask_one(i, T.size(0) + P.size(0)),
            'phase': torch.tensor(1)
        })

        while q_reset == 0 and P[q + 1].item() != T[i].item():
            if is_reset[q] == 1:
                q = 0
                q_reset = 1
            else:
                q = pi[q].item()

            data.increase_hints({
                'pred_h': strings_pred(T_pos, P_pos),
                'pi': strings_pi(T_pos, P_pos, pi),
                'is_reset': torch.concatenate([torch.zeros(T.size(0)), is_reset.clone()]),
                'k': mask_one(T.size(0) + k, T.size(0) + P.size(0)),
                'k_reset': torch.tensor(k_reset),
                'q': mask_one(T.size(0) + q, T.size(0) + P.size(0)),
                'q_reset': torch.tensor(q_reset),
                's': mask_one(s, T.size(0) + P.size(0)),
                'i': mask_one(i, T.size(0) + P.size(0)),
                'phase': torch.tensor(1)
          })
            
        if q_reset == 1:
            q = -1
            q_reset = 0
        
        if P[q + 1].item() == T[i].item():
            if q == P.size(0) - 2:
                data.set_outputs({
                    'match': mask_one(s, T.size(0) + P.size(0))
                })
        
                return data
            q += 1
        
        if q == -1:
            q_reset = 1
            q = 0

    data.set_outputs({
        'match': mask_one(T.size(0), T.size(0) + P.size(0))
    })
        
    return data
