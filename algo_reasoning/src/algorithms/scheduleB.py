import itertools
import os
import torch
from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type

schedule_specs = {
    'N': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'W': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    'C': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR)
}

def schedule(N, W, nb_nodes):
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
        data_point["max_length"] = max_length
        torch.save(data_point, f"tmp/CLRS30/schedule/train/{i}")

    for i, data_point in enumerate(val_datapoints):
        data_point["max_length"] = max_length
        torch.save(data_point, f"tmp/CLRS30/schedule/val/{i}")

    for i, data_point in enumerate(test_datapoints):
        data_point["max_length"] = max_length
        torch.save(data_point, f"tmp/CLRS30/schedule/test/{i}")