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
    inputs['N'] = torch.tensor([N])
    inputs['W'] = torch.tensor([W])
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
    outputs['C'] = torch.tensor([c])

    return CLRSData(inputs=inputs, hints=torch.tensor([]), length=torch.tensor(length).float(), outputs=outputs, algorithm="schedule")

if __name__ == "__main__":
    os.mkdir("tmp/CLRS30/schedule")
    
    os.mkdir("tmp/CLRS30/schedule/train")

    # Sampling Training set
    N_train = torch.randint(2, 10**2 + 1, (1000,)).tolist()
    W_train = torch.randint(1, 53, (1000,)).tolist()

    for i, (N, W) in enumerate(zip(N_train, W_train)):
        print(i)
        data_point = schedule(N, W, 16)
        torch.save(data_point, f"tmp/CLRS30/schedule/train/{i}")

    os.mkdir("tmp/CLRS30/schedule/val")
    # Sampling Validation set
    N_val = torch.randint(2, 10**2 + 1, (32,)).tolist()
    W_val = torch.randint(1, 53, (32,)).tolist()

    for i, (N, W) in enumerate(zip(N_val, W_val)):
        print(i)
        data_point = schedule(N, W, 16)
        torch.save(data_point, f"tmp/CLRS30/schedule/val/{i}")


    os.mkdir("tmp/CLRS30/schedule/test")
    # Sampling Test set
    N_test = torch.randint(2, 10**2 + 1, (32,)).tolist()
    W_test = torch.randint(1, 53, (32,)).tolist()

    for i, (N, W) in enumerate(zip(N_test, W_test)):
        print(i)
        data_point = schedule(N, W, 64)
        torch.save(data_point, f"tmp/CLRS30/schedule/test/{i}")
