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

"""Geometry algorithm generators.

Currently implements the following:
- Segment intersection
- Graham scan convex hull (Graham, 1972)
- Jarvis' march convex hull (Jarvis, 1973)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# For the original implementation of such algorithms, please refer to: https://github.com/google-deepmind/clrs

# This implementation is pretty much the same as the original one, 
# but with some minor changes in the way the data is handled to better fit the Dataset in torch framework format.
# As the original implementation is made in the Haiku framework, which is a JAX-based framework,

# Modifications were made by Ronald Albert (https://www.pesc.coppe.ufrj.br/index.php/pt-BR/pessoas/details/18/2955), throughout the course of his Master's degree in Computer Science 
# at the Federal University of Rio de Janeiro (UFRJ), Brazil.

import torch
import math

from algo_reasoning.src.data import AlgorithmicData
from algo_reasoning.src.probing import probe_array, mask_one

def segments_intersect(xs, ys, nb_nodes, *args, **kwargs):
    """Segment intersection."""

    data = AlgorithmicData(algorithm="segments_intersect", *args, **kwargs)

    dirs = torch.zeros(xs.size(0))
    on_seg = torch.zeros(xs.size(0))

    data.set_inputs({
        'x': xs.clone(),
        'y': ys.clone()
    }, nb_nodes)

    data.increase_hints({
        'i': mask_one(0, xs.size(0)),
        'j': mask_one(0, xs.size(0)),
        'k': mask_one(0, xs.size(0)),
        'dir': dirs.clone(),
        'on_seg': on_seg.clone()
    })

    def cross_product(x1, y1, x2, y2):
        return x1.item() * y2.item() - x2.item() * y1.item()

    def direction(xs, ys, i, j, k):
        return cross_product(xs[k] - xs[i], ys[k] - ys[i], xs[j] - xs[i], ys[j] - ys[i])

    def on_segment(xs, ys, i, j, k):
        if min(xs[i], xs[j]) <= xs[k] and xs[k] <= max(xs[i], xs[j]):
            if min(ys[i], ys[j]) <= ys[k] and ys[k] <= max(ys[i], ys[j]):
                return 1
        return 0

    dirs[0] = direction(xs, ys, 2, 3, 0)
    on_seg[0] = on_segment(xs, ys, 2, 3, 0)

    data.increase_hints({
        'i': mask_one(2, xs.size(0)),
        'j': mask_one(3, xs.size(0)),
        'k': mask_one(0, xs.size(0)),
        'dir': dirs.clone(),
        'on_seg': on_seg.clone()
    })

    dirs[1] = direction(xs, ys, 2, 3, 1)
    on_seg[1] = on_segment(xs, ys, 2, 3, 1)

    data.increase_hints({
        'i': mask_one(2, xs.size(0)),
        'j': mask_one(3, xs.size(0)),
        'k': mask_one(1, xs.size(0)),
        'dir': dirs.clone(),
        'on_seg': on_seg.clone()
    })

    dirs[2] = direction(xs, ys, 0, 1, 2)
    on_seg[2] = on_segment(xs, ys, 0, 1, 2)

    data.increase_hints({
        'i': mask_one(0, xs.size(0)),
        'j': mask_one(1, xs.size(0)),
        'k': mask_one(2, xs.size(0)),
        'dir': dirs.clone(),
        'on_seg': on_seg.clone()
    })

    dirs[3] = direction(xs, ys, 0, 1, 3)
    on_seg[3] = on_segment(xs, ys, 0, 1, 3)

    data.increase_hints({
        'i': mask_one(0, xs.size(0)),
        'j': mask_one(1, xs.size(0)),
        'k': mask_one(3, xs.size(0)),
        'dir': dirs.clone(),
        'on_seg': on_seg.clone()
    })

    ret = 0

    if ((dirs[0].item() > 0 and dirs[1].item() < 0) or (dirs[0].item() < 0 and dirs[1].item() > 0)) and (
       (dirs[2].item() > 0 and dirs[3].item() < 0) or (dirs[2].item() < 0 and dirs[3].item() > 0)):
        ret = 1
    elif dirs[0].item() == 0 and on_seg[0].item():
        ret = 1
    elif dirs[1].item() == 0 and on_seg[1].item():
        ret = 1
    elif dirs[2].item() == 0 and on_seg[2].item():
        ret = 1
    elif dirs[3].item() == 0 and on_seg[3].item():
        ret = 1

    data.set_outputs({
       'intersect': torch.tensor(ret)
    })

    return data


def graham_scan(xs, ys, nb_nodes, *args, **kwargs):
    """Graham scan convex hull (Graham, 1972)."""

    data = AlgorithmicData(algorithm="graham_scan", *args, **kwargs)
    data.set_inputs({
        'x': xs.clone(),
        'y': ys.clone()
    }, nb_nodes)

    A_pos = torch.arange(xs.size(0))
    in_hull = torch.zeros(xs.size(0))
    stack_prev = torch.arange(xs.size(0))
    atans = torch.zeros(xs.size(0))

    data.increase_hints({
        'best': mask_one(0, xs.size(0)),
        'atans': atans.clone(),
        'in_hull_h': in_hull.clone(),
        'stack_prev': stack_prev.clone(),
        'last_stack': mask_one(0, xs.size(0)),
        'i': mask_one(0, xs.size(0)),
        'phase': mask_one(0, 5)
    })

    def counter_clockwise(xs, ys, i, j, k):
        return ((xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) * (xs[k] - xs[i])).item() <= 0

    best = 0
    for i in range(xs.size(0)):
        if ys[i] < ys[best] or (ys[i] == ys[best] and xs[i] < xs[best]):
            best = i

    in_hull[best] = 1
    last_stack = best

    data.increase_hints({
          'best': mask_one(best, xs.size(0)),
          'atans': atans.clone(),
          'in_hull_h': in_hull.clone(),
          'stack_prev': stack_prev.clone(),
          'last_stack': mask_one(last_stack, xs.size(0)),
          'i': mask_one(best, xs.size(0)),
          'phase': mask_one(1, 5)
    })


    for i in range(xs.shape[0]):
        if i != best:
            atans[i] = math.atan2((ys[i] - ys[best]).item(), (xs[i] - xs[best]).item())
    
    atans[best] = -123456789
    ind = torch.argsort(atans)
    atans[best] = 0

    data.increase_hints({
          'best': mask_one(best, xs.size(0)),
          'atans': atans.clone(),
          'in_hull_h': in_hull.clone(),
          'stack_prev': stack_prev.clone(),
          'last_stack': mask_one(last_stack, xs.size(0)),
          'i': mask_one(best, xs.size(0)),
          'phase': mask_one(2, 5)
    })

    for i in range(1, xs.size(0)):
        if i >= 3:
            while counter_clockwise(xs, ys, stack_prev[last_stack], last_stack, ind[i]):
                prev_last = last_stack
                last_stack = stack_prev[last_stack].item()
                stack_prev[prev_last] = prev_last
                in_hull[prev_last] = 0

                data.increase_hints({
                    'best': mask_one(best, xs.size(0)),
                    'atans': atans.clone(),
                    'in_hull_h': in_hull.clone(),
                    'stack_prev': stack_prev.clone(),
                    'last_stack': mask_one(last_stack, xs.size(0)),
                    'i': mask_one(A_pos[ind[i]], xs.size(0)),
                    'phase': mask_one(3, 5)
                })

        in_hull[ind[i]] = 1
        stack_prev[ind[i]] = last_stack
        last_stack = ind[i].item()

        data.increase_hints({
                'best': mask_one(best, xs.size(0)),
                'atans': atans.clone(),
                'in_hull_h': in_hull.clone(),
                'stack_prev': stack_prev.clone(),
                'last_stack': mask_one(last_stack, xs.size(0)),
                'i': mask_one(A_pos[ind[i]], xs.size(0)),
                'phase': mask_one(4, 5)
        })
    
    data.set_outputs({
        'in_hull': in_hull.clone()
    })

    return data


def jarvis_march(xs, ys, nb_nodes, *args, **kwargs):
    """Jarvis' march convex hull (Jarvis, 1973)."""
    
    data = AlgorithmicData(algorithm="jarvis_march", *args, **kwargs)
    data.set_inputs({
        'x': xs.clone(),
        'y': ys.clone()
    }, nb_nodes)
  
    A_pos = torch.arange(xs.size(0))
    in_hull = torch.zeros(xs.size(0))

    data.increase_hints({
          'pred_h': probe_array(A_pos.clone()),
          'in_hull_h': in_hull.clone(),
          'best': mask_one(0, xs.size(0)),
          'last_point': mask_one(0, xs.size(0)),
          'endpoint': mask_one(0, xs.size(0)),
          'i': mask_one(0, xs.size(0)),
          'phase': mask_one(0, 2)
    })

    def counter_clockwise(xs, ys, i, j, k):
        if (k == i) or (k == j):
            return False
        return ((xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) * (xs[k] - xs[i])).item() <= 0

    best = 0
    for i in range(xs.size(0)):
        if ys[i] < ys[best] or (ys[i] == ys[best] and xs[i] < xs[best]):
            best = i

    in_hull[best] = 1
    last_point = best
    endpoint = 0

    data.increase_hints({
          'pred_h': probe_array(A_pos.clone()),
          'in_hull_h': in_hull.clone(),
          'best': mask_one(best, xs.size(0)),
          'last_point': mask_one(last_point, xs.size(0)),
          'endpoint': mask_one(endpoint, xs.size(0)),
          'i': mask_one(0, xs.size(0)),
          'phase': mask_one(1, 2)
      })


    while True:
        for i in range(xs.size(0)):
            if endpoint == last_point or counter_clockwise(xs, ys, last_point, endpoint, i):
                endpoint = i

            data.increase_hints({
              'pred_h': probe_array(A_pos.clone()),
              'in_hull_h': in_hull.clone(),
              'best': mask_one(best, xs.size(0)),
              'last_point': mask_one(last_point, xs.size(0)),
              'endpoint': mask_one(endpoint, xs.size(0)),
              'i': mask_one(i, xs.size(0)),
              'phase': mask_one(1, 2)
          })
      
        if in_hull[endpoint] > 0:
            break

        in_hull[endpoint] = 1
        last_point = endpoint
        endpoint = 0

        data.increase_hints({
            'pred_h': probe_array(A_pos.clone()),
            'in_hull_h': in_hull.clone(),
            'best': mask_one(best, xs.size(0)),
            'last_point': mask_one(last_point, xs.size(0)),
            'endpoint': mask_one(endpoint, xs.size(0)),
            'i': mask_one(0, xs.size(0)),
            'phase': mask_one(1, 2)
        })

    data.set_outputs({
        'in_hull': in_hull.clone()
    })


    return data
