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
    'output_score_D1': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
    'score_D1': (Stage.HINT, Location.NODE, Type.SCALAR),
    'score_D2': (Stage.HINT, Location.NODE, Type.SCALAR),
    "in_hull": (Stage.HINT, Location.NODE, Type.MASK)
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

    ordering = [best]
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
        ordering.append(endpoint)
        last_point = endpoint
        endpoint = 0

    return in_hull, ordering


def three_kinds_dice(N_faces1, N_faces2, values_D1, values_D2, nb_nodes):
    inputs = CLRSData()
    inputs['pos'] = ((torch.arange(nb_nodes) * 1.0)/nb_nodes).unsqueeze(0)
    inputs['N_faces1'] = torch.tensor([N_faces1]).float()
    inputs['N_faces2'] = torch.tensor([N_faces2]).float()

    inputs['values_D1'] = torch.bincount(values_D1, minlength=nb_nodes).float().unsqueeze(0)
    inputs['values_D2'] = torch.bincount(values_D2, minlength=nb_nodes).float().unsqueeze(0)

    max_value = torch.max(torch.concat((values_D1, values_D2))).item() + 1
    min_value = torch.min(torch.concat((values_D1, values_D2))).item() - 1

    length = 1

    hints = CLRSData() 
    score_D1 = torch.tensor([(torch.sum(v > values_D1) + torch.sum(v == values_D1)/2).item()/values_D1.size(0) for v in torch.arange(start=1, end=nb_nodes + 1)])
    score_D2 = torch.tensor([(torch.sum(v > values_D2) + torch.sum(v == values_D2)/2).item()/values_D2.size(0) for v in torch.arange(start=1, end=nb_nodes + 1)])
    hints['score_D1'] = score_D1.float().unsqueeze(0).unsqueeze(0)
    hints['score_D2'] = score_D2.float().unsqueeze(0).unsqueeze(0)

    in_hull_output, ordering = jarvis_march(score_D1[min_value:max_value], score_D2[min_value:max_value])
    in_hull = torch.zeros(score_D1.shape)
    in_hull[min_value:max_value] = in_hull_output

    print(ordering)

    hints['in_hull'] = in_hull.unsqueeze(0).unsqueeze(0)

    output_score_D1 = 0
    output_score_D2 = 1
    print(score_D1)
    print(score_D2)
    score_D1_in_hull = score_D1[min_value:max_value][ordering]
    score_D2_in_hull = score_D2[min_value:max_value][ordering]

    print(score_D1_in_hull)
    print(score_D2_in_hull)

    n_hull_points = in_hull.sum().long().item()
    for i in range(n_hull_points):
        x1 = score_D1_in_hull[i].item()
        x2 =  score_D1_in_hull[(i + 1) % n_hull_points].item()

        y1 = score_D2_in_hull[i].item()
        y2 = score_D2_in_hull[(i + 1) % n_hull_points].item()

        if ((x1 <= 0.5) and (0.5 <= x2)) or ((x2 <= 0.5) and (0.5 <= x1)):
            t = (0.5 - x1) / (x2 - x1)
            y_intersect = y1 + t * (y2 - y1)

            print("Y intersect: ", y_intersect)

            output_score_D1 = max(output_score_D1, y_intersect)

        if ((y1 <= 0.5) and (0.5 <= y2)) or ((y2 <= 0.5) and (0.5 <= y1)):
            t = (0.5 - y1) / (y2 - y1)
            x_intersect = x1 + t * (x2 - x1)

            print("X intersect: ", x_intersect)

            output_score_D2 = min(output_score_D2, x_intersect)

    outputs = CLRSData()
    outputs['output_score_D1'] = torch.tensor([output_score_D1]).float()
    outputs['output_score_D2'] = torch.tensor([output_score_D2]).float()

    return CLRSData(inputs=inputs, hints=hints, length=torch.tensor(length).float(), outputs=outputs, algorithm="three_kinds_dice")
    

if __name__ == "__main__":
    os.mkdir("tmp/CLRS30/three_kinds_dice")
    
    os.mkdir("tmp/CLRS30/three_kinds_dice/train")

    # Sampling Training set
    N_faces1_train = torch.randint(1, 100, (1000,)).tolist()
    N_faces2_train = torch.randint(1, 100, (1000,)).tolist()

    train_datapoints = []
    max_length = -1
    for N_faces1, N_faces2 in zip(N_faces1_train, N_faces2_train):
        nb_nodes = 16
        values_D1 = torch.randint(1, nb_nodes, (N_faces1, ))
        values_D2 = torch.randint(1, nb_nodes, (N_faces1, ))

        data_point = three_kinds_dice(N_faces1, N_faces2, values_D1, values_D2, nb_nodes)
        train_datapoints.append(data_point)
        curr_length = data_point.length.long().item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/three_kinds_dice/val")
    val_datapoints = []
    # Sampling Validation set
    N_faces1_val = torch.randint(1, 100, (32,)).tolist()
    N_faces2_val = torch.randint(1, 100, (32,)).tolist()

    for N_faces1, N_faces2 in zip(N_faces1_val, N_faces2_val):
        nb_nodes = 16
        values_D1 = torch.randint(1, nb_nodes, (N_faces1, ))
        values_D2 = torch.randint(1, nb_nodes, (N_faces1, ))

        data_point = three_kinds_dice(N_faces1, N_faces2, values_D1, values_D2, nb_nodes)
        val_datapoints.append(data_point)
        curr_length = data_point.length.long().item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/three_kinds_dice/test")
    test_datapoints = []
    # Sampling Test set
    N_faces1_test = torch.randint(1, 100, (32,)).tolist()
    N_faces2_test  = torch.randint(1, 100, (32,)).tolist()

    for N_faces1, N_faces2 in zip(N_faces1_test, N_faces2_test):
        nb_nodes = 64
        values_D1 = torch.randint(1, nb_nodes, (N_faces1, ))
        values_D2 = torch.randint(1, nb_nodes, (N_faces1, ))

        data_point = three_kinds_dice(N_faces1, N_faces2, values_D1, values_D2, nb_nodes)
        test_datapoints.append(data_point)
        curr_length = data_point.length.long().item()
        max_length = curr_length if curr_length > max_length else max_length

    for i, data_point in enumerate(train_datapoints):
        data_point["max_length"] = max_length
        torch.save(data_point, f"tmp/CLRS30/three_kinds_dice/train/{i}")

    for i, data_point in enumerate(val_datapoints):
        data_point["max_length"] = max_length
        torch.save(data_point, f"tmp/CLRS30/three_kinds_dice/val/{i}")

    for i, data_point in enumerate(test_datapoints):
        data_point["max_length"] = max_length
        torch.save(data_point, f"tmp/CLRS30/three_kinds_dice/test/{i}")