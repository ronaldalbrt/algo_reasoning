import itertools
import os
import torch
from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type

three_kinds_dice_specs = {
    'N_faces1': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'N_faces2': (Stage.INPUT, Location.NODE, Type.SCALAR),
    
}

def three_kinds_dice(N, W, nb_nodes):
    inputs = CLRSData()
    inputs['N'] = torch.tensor([N]).float()
    inputs['W'] = torch.tensor([W]).float()
    inputs['pos'] = ((torch.arange(nb_nodes) * 1.0)/nb_nodes).unsqueeze(0)
    length = 0
    c = 4

    while c <= W:
        length += 1
        sch = []
        cur = [1] * c
        for i in range(c // 2, c):
            cur[i] = 2
        perm = sorted(itertools.permutations(cur))

        for p in perm:
            sch.append(list(p))

        # Remove duplicates and keep order
        sch = list(k for k, _ in itertools.groupby(sch))

        if len(sch) >= N:
            break
        c += 1

    if c > W:
        c = -1

    # for i in range(W):
    #     for j in range(N):
    #         print(sch[j][i % c], end="")
    #     print()

    outputs = CLRSData()
    outputs['C'] = torch.tensor([c]).float()

    return CLRSData(inputs=inputs, hints=CLRSData(), length=torch.tensor(length).float(), outputs=outputs, algorithm="schedule")