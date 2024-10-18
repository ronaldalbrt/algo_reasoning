import torch
from sympy.utilities.iterables import multiset_permutations

from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type

schedule_specs = {
    'n': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'w': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    'c': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
    'infinity':(Stage.OUTPUT, Location.GRAPH, Type.MASK),
    'c_h': (Stage.HINT, Location.GRAPH, Type.SCALAR)
}

def schedule(N, W, nb_nodes, *args, **kwargs):
    data = CLRSData(algorithm="schedule", *args, **kwargs)
    
    data = data.set_inputs({
        'n': torch.tensor(N),
        'w': torch.tensor(W)
    }, nb_nodes)

    infinity = 0
    c = 4
    data = data.increase_hints({'c_h': torch.tensor(c)})

    while c <= W:
        
        sch = []
        cur = ([1] * (c//2)) + ([2] * (c  - c//2))

        for perm in multiset_permutations(cur):
            if perm[0] != 1: break
            
            sch.append(list(perm))

        if len(sch) >= N:
            break

        c += 1
        data = data.increase_hints({'c_h': torch.tensor(c)})

    if c > W:
        infinity = 1
        c = -1

    data = data.set_outputs({
        'c': torch.tensor(c),
        'infinity': torch.tensor(infinity)
    })

    return data