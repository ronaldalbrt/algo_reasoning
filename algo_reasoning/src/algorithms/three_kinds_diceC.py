import itertools
import os
import torch
import numpy as np
import math
from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type

three_kinds_dice_specs = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    'N_faces1': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'N_faces2': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'values_D1': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'values_D2': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'score_D1': (Stage.HINT, Location.NODE, Type.SCALAR),
    'score_D2': (Stage.HINT, Location.NODE, Type.SCALAR),
}

def graham_scan(xs, ys):
    """Graham scan convex hull (Graham, 1972)."""
    in_hull = torch.zeros(xs.shape[0])
    stack_prev = torch.arange(xs.shape[0])
    atans = torch.zeros(xs.shape[0])

    def counter_clockwise(xs, ys, i, j, k):
        return ((xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) * (xs[k] - xs[i])) <= 0

    best = 0
    for i in range(xs.shape[0]):
        if ys[i] < ys[best] or (ys[i] == ys[best] and xs[i] < xs[best]):
            best = i

    in_hull[best] = 1
    last_stack = best

    for i in range(xs.shape[0]):
        if i != best:
            atans[i] = math.atan2(ys[i] - ys[best], xs[i] - xs[best])
    atans[best] = -123456789
    ind = torch.argsort(atans)
    atans[best] = 0

    for i in range(1, xs.shape[0]):
        if i >= 3:
            while counter_clockwise(xs, ys, stack_prev[last_stack], last_stack, ind[i]):
                prev_last = last_stack
                last_stack = stack_prev[last_stack]
                stack_prev[prev_last] = prev_last
                in_hull[prev_last] = 0

    in_hull[ind[i]] = 1
    stack_prev[ind[i]] = last_stack
    last_stack = ind[i]

    return in_hull

def jarvis_march(xs, ys):
    """Jarvis' march convex hull (Jarvis, 1973)."""
    in_hull = torch.zeros(xs.shape[0])

    def counter_clockwise(xs, ys, i, j, k):
        if (k == i) or (k == j):
            return False
        
        return ((xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) *
                (xs[k] - xs[i])) <= 0

    best = 0
    for i in range(xs.shape[0]):
        if ys[i] < ys[best] or (ys[i] == ys[best] and xs[i] < xs[best]):
            best = i

    in_hull[best] = 1
    last_point = best
    endpoint = 0

    while True:
        for i in range(xs.shape[0]):
            if endpoint == last_point or counter_clockwise(xs, ys, last_point, endpoint, i):
                endpoint = i

        if in_hull[endpoint] > 0:
            break
        in_hull[endpoint] = 1
        last_point = endpoint
        endpoint = 0

    return in_hull


def three_kinds_dice(N_faces1, N_faces2, values_D1, values_D2, nb_nodes):
    inputs = CLRSData()
    inputs['pos'] = ((torch.arange(nb_nodes) * 1.0)/nb_nodes).unsqueeze(0)
    inputs['N_faces1'] = torch.tensor([N_faces1]).float()
    inputs['N_faces2'] = torch.tensor([N_faces2]).float()

    inputs['values_D1'] = torch.bincount(values_D1, minlength=nb_nodes).float().unsqueeze(0)
    inputs['values_D2'] = torch.bincount(values_D2, minlength=nb_nodes).float().unsqueeze(0)

    length = 0

    hints = CLRSData() 
    score_D1 = [(torch.sum(v > values_D1) + torch.sum(v == values_D1)/2).item()/values_D1.size(0) for v in torch.arange(start=1,end=nb_nodes+1)]
    score_D2 = [(torch.sum(v > values_D2) + torch.sum(v == values_D2)/2).item()/values_D2.size(0) for v in torch.arange(start=1,end=nb_nodes+1)]
    hints['score_D1'] = torch.tensor(score_D1).float().unsqueeze(0)
    hints['score_D2'] = torch.tensor(score_D2).float().unsqueeze(0)

    in_hull = jarvis_march(torch.tensor(score_D1), torch.tensor(score_D2))
    hints['in_hull'] = in_hull.unsqueeze(0)

    return CLRSData(inputs=inputs, hints=hints, length=torch.tensor(length).float(), outputs=outputs, algorithm="three_kinds_dice")
 

