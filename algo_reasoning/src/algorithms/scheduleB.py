from itertools import permutations, groupby
from sympy.utilities.iterables import multiset_permutations
import os
import torch
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

def schedule(N, W, nb_nodes):
    data = CLRSData()
    data.set_inputs({
        'n': torch.tensor([N]).float(),
        'w': torch.tensor([W]).float(),
        'pos': ((torch.arange(nb_nodes) * 1.0)/nb_nodes).unsqueeze(0)
    })

    infinity = 0
    c = 4
    data.increase_hints({'c_h': torch.tensor([c]).float().unsqueeze(0)})

    while c <= W:
        
        sch = []
        cur = ([1] * (c//2)) + ([2] * (c  - c//2))

        for perm in multiset_permutations(cur):
            if perm[0] != 1: break
            
            sch.append(list(perm))

        if len(sch) >= N:
            break

        c += 1
        data.increase_hints({'c_h': torch.tensor([c]).float().unsqueeze(0)})

    if c > W:
        infinity = 1
        c = -1

    data.set_outputs({
        'c': torch.tensor([c]).float(),
        'infinity': torch.tensor([infinity]).float()
    })

    return data

if __name__ == "__main__":
    os.mkdir("tmp/CLRS30/schedule")
    
    os.mkdir("tmp/CLRS30/schedule/train")

    # Sampling Training set
    N_train = torch.randint(2, 10**2 + 1, (1000,)).tolist()
    W_train = torch.randint(1, 53, (1000,)).tolist()

    train_datapoints = []
    max_length = -1
    for N, W in zip(N_train, W_train):
        data_point = schedule(N, W, 16)
        train_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/schedule/val")
    val_datapoints = []
    # Sampling Validation set
    N_val = torch.randint(2, 10**2 + 1, (32,)).tolist()
    W_val = torch.randint(1, 53, (32,)).tolist()

    for N, W in zip(N_val, W_val):
        data_point = schedule(N, W, 16)
        val_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/schedule/test")
    test_datapoints = []
    # Sampling Test set
    N_test = torch.randint(2, 10**2 + 1, (32,)).tolist()
    W_test = torch.randint(1, 53, (32,)).tolist()

    for N, W in zip(N_test, W_test):
        data_point = schedule(N, W, 64)
        test_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length

    for i, data_point in enumerate(train_datapoints):
        torch.save(data_point, f"tmp/CLRS30/schedule/train/{i}")

    for i, data_point in enumerate(val_datapoints):
        torch.save(data_point, f"tmp/CLRS30/schedule/val/{i}")

    for i, data_point in enumerate(test_datapoints):
        torch.save(data_point, f"tmp/CLRS30/schedule/test/{i}")