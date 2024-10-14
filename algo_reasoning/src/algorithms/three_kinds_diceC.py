import os
import torch
from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type
from random import sample

# GEOMETRY / CONVEX HULL

three_kinds_dice_specs = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    'values_D1': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'values_D2': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'output_score_D1': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
    'output_score_D2': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
    'score_D1': (Stage.HINT, Location.NODE, Type.SCALAR),
    'score_D2': (Stage.HINT, Location.NODE, Type.SCALAR),
    "in_hull": (Stage.HINT, Location.NODE, Type.MASK)
}

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

def three_kinds_dice(values_D1, values_D2, nb_nodes):
    inputs = CLRSData()
    inputs['pos'] = ((torch.arange(nb_nodes) * 1.0)/nb_nodes).unsqueeze(0)

    inputs['values_D1'] = torch.bincount(values_D1, minlength=nb_nodes).float().unsqueeze(0)
    inputs['values_D2'] = torch.bincount(values_D2, minlength=nb_nodes).float().unsqueeze(0)

    max_value = torch.max(torch.concat((values_D1, values_D2))).item() + 1
    min_value = torch.min(torch.concat((values_D1, values_D2))).item() - 2
    min_value = 0 if min_value < 0 else min_value

    length = 1

    hints = CLRSData() 
    score_D1 = torch.tensor([(torch.sum(v > values_D1) + torch.sum(v == values_D1)/2).item()/values_D1.size(0) for v in torch.arange(start=1, end=nb_nodes + 1)])
    score_D2 = torch.tensor([(torch.sum(v > values_D2) + torch.sum(v == values_D2)/2).item()/values_D2.size(0) for v in torch.arange(start=1, end=nb_nodes + 1)])
    hints['score_D1'] = score_D1.float().unsqueeze(0).unsqueeze(0)
    hints['score_D2'] = score_D2.float().unsqueeze(0).unsqueeze(0)
    hints['in_hull'] = torch.zeros(nb_nodes).unsqueeze(0).unsqueeze(0)

    in_hull_output = jarvis_march(score_D1[min_value:max_value], score_D2[min_value:max_value])
    in_hull = torch.zeros(nb_nodes)
    in_hull[min_value:max_value] = in_hull_output

    length += 1
    hints['score_D1'] = torch.cat((hints['score_D1'], score_D1.float().unsqueeze(0).unsqueeze(0)), 1)
    hints['score_D2'] = torch.cat((hints['score_D2'], score_D2.float().unsqueeze(0).unsqueeze(0)), 1)
    hints['in_hull'] = torch.cat((hints['in_hull'], in_hull.unsqueeze(0).unsqueeze(0)), 1)

    output_score_D1 = 0
    output_score_D2 = 1
    score_D1_in_hull = score_D1[min_value:max_value][in_hull_output.bool()]
    score_D2_in_hull = score_D2[min_value:max_value][in_hull_output.bool()]

    scores = torch.stack((score_D2_in_hull, score_D1_in_hull), dim=1)
    
    for rep in range(2):
        hull = torch.tensor([], dtype=torch.int32)

        for i in range(scores.size(0)):
            x3, y3 = scores[i]
            while hull.size(0) >= 2:
                x1, y1 = scores[hull[-2]]
                x2, y2 = scores[hull[-1]]

                if ((x3 - x1) * (y2 - y1) < (x2 - x1) * (y3 - y1)).item():
                    break

                hull = hull[:-1]

            hull = torch.cat((hull, torch.tensor([i], dtype=torch.int32)))

        ans = 1
        scores = scores[hull]

        for i in range(scores.size(0)):
            x1, y1 = scores[i - 1]
            x2, y2 = scores[i]

            if x1 >= 0.5 or x2 < 0.5:
                continue

            ans = y1 + (y2 - y1) / (x2 - x1) * (0.5 - x1)

        if rep == 0:
            output_score_D2 = ans
        else:
            output_score_D1 = 1 - ans

        scores = torch.flip(scores, dims=[0, 1])
        scores = 1 - scores

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
        nb_nodes = sample([4, 7, 11, 13, 16], 1)[0]
        values_D1 = torch.randint(1, nb_nodes, (N_faces1, ))
        values_D2 = torch.randint(1, nb_nodes, (N_faces2, ))

        data_point = three_kinds_dice(values_D1, values_D2, nb_nodes)
        train_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/three_kinds_dice/val")
    val_datapoints = []
    # Sampling Validation set
    N_faces1_val = torch.randint(1, 100, (32,)).tolist()
    N_faces2_val = torch.randint(1, 100, (32,)).tolist()

    for N_faces1, N_faces2 in zip(N_faces1_val, N_faces2_val):
        nb_nodes = 16
        values_D1 = torch.randint(1, nb_nodes, (N_faces1, ))
        values_D2 = torch.randint(1, nb_nodes, (N_faces2, ))

        data_point = three_kinds_dice(values_D1, values_D2, nb_nodes)
        val_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/three_kinds_dice/test")
    test_datapoints = []
    # Sampling Test set
    N_faces1_test = torch.randint(1, 100, (32,)).tolist()
    N_faces2_test  = torch.randint(1, 100, (32,)).tolist()

    for N_faces1, N_faces2 in zip(N_faces1_test, N_faces2_test):
        nb_nodes = 64
        values_D1 = torch.randint(1, nb_nodes, (N_faces1, ))
        values_D2 = torch.randint(1, nb_nodes, (N_faces2, ))

        data_point = three_kinds_dice(values_D1, values_D2, nb_nodes)
        test_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length

    for i, data_point in enumerate(train_datapoints):
        torch.save(data_point, f"tmp/CLRS30/three_kinds_dice/train/{i}")

    for i, data_point in enumerate(val_datapoints):
        torch.save(data_point, f"tmp/CLRS30/three_kinds_dice/val/{i}")

    for i, data_point in enumerate(test_datapoints):
        torch.save(data_point, f"tmp/CLRS30/three_kinds_dice/test/{i}")