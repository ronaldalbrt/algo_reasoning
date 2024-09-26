import os
import torch
import torch.linalg as LA
import math
from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type

waterworld_specs = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    'n': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'm': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'area_percentages': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'surface_percentage':(Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
    'total_area': (Stage.HINT, Location.GRAPH, Type.SCALAR),
    'area_sums': (Stage.HINT, Location.GRAPH, Type.SCALAR)
}

def waterworld(n, m, area_percentages, nb_nodes):
    inputs = CLRSData()
    inputs['pos'] = ((torch.arange(nb_nodes) * 1.0)/nb_nodes).unsqueeze(0)
    
    inputs['n'] = torch.tensor([n]).float()
    inputs['m'] = torch.tensor([m]).float()

    inputs['area_percentages'] = area_percentages.unsqueeze(0)
    length = 0

    hints = CLRSData()
    area_sums = 0
    total_area = n*m

    hints["total_area"] = torch.tensor([total_area]).unsqueeze(0).float()
    hints["area_sums"] = torch.tensor([area_sums]).unsqueeze(0).float()
    for i in range(n*m):
        length += 1

        area_sums += area_percentages[i].item()
        hints['total_area'] = torch.cat((hints['total_area'], torch.tensor([total_area]).unsqueeze(0).float()), 1)
        hints['area_sums'] = torch.cat((hints['area_sums'],  torch.tensor([area_sums]).unsqueeze(0).float()), 1)

    surface_percentage = area_sums/total_area
    outputs = CLRSData()
    outputs['surface_percentage'] = torch.tensor([surface_percentage]).float()

    return CLRSData(inputs=inputs, hints=hints, length=torch.tensor(length).float(), outputs=outputs, algorithm="waterworld")

if __name__ == "__main__":
    os.mkdir("tmp/CLRS30/waterworld")
    os.mkdir("tmp/CLRS30/waterworld/train")

    # Sampling Training set
    train_datapoints = []
    max_length = -1
    for _ in range(1000):
        n, m  = (v.item() for v in torch.randint(1, 9, (2,)))
        nb_nodes = n*m

        ap = torch.randint(101, (nb_nodes,))

        data_point = waterworld(n, m, ap, nb_nodes)
        train_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/waterworld/val")

    # Sampling Validation set
    val_datapoints = []
    max_length = -1
    for _ in range(32):
        n, m  = (v.item() for v in torch.randint(1, 9, (2,)))
        nb_nodes = n*m

        ap = torch.randint(101, (nb_nodes,))

        data_point = waterworld(n, m, ap, nb_nodes)
        val_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/waterworld/test")
    test_datapoints = []
    # Sampling Test set
    max_length = -1
    for _ in range(32):
        n, m  = (v.item() for v in torch.randint(1, 9, (2,)))
        nb_nodes = n*m

        ap = torch.randint(101, (nb_nodes,))

        data_point = waterworld(n, m, ap, nb_nodes)
        test_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length

    for i, data_point in enumerate(train_datapoints):
        data_point["max_length"] = torch.tensor(max_length)
        torch.save(data_point, f"tmp/CLRS30/waterworld/train/{i}")

    for i, data_point in enumerate(val_datapoints):
        data_point["max_length"] = torch.tensor(max_length)
        torch.save(data_point, f"tmp/CLRS30/waterworld/val/{i}")

    for i, data_point in enumerate(test_datapoints):
        data_point["max_length"] = torch.tensor(max_length)
        torch.save(data_point, f"tmp/CLRS30/waterworld/test/{i}")