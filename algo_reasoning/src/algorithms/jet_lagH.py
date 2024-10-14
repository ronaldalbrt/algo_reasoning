import os
import torch
import torch.linalg as LA
import math
from algo_reasoning.src.data import CLRSData
from algo_reasoning.src.specs import Stage, Location, Type

jet_lag_specs = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    'b': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'e': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'n': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'impossible': (Stage.OUTPUT, Location.GRAPH, Type.MASK),
    }

def sample_tasks(nb_nodes, max_activity_dur, max_interval):
    last_e = 0
    b = torch.tensor([])
    e = torch.tensor([])

    for _ in range(nb_nodes):
        last_b = last_e + torch.randint(max_interval, (1,))

        b = torch.concat((b, last_b), dim=0)
        e = torch.concat((e, last_b + torch.randint(max_activity_dur, (1,))), dim=0)

        last_e = e[-1].item()

    return b, e

def jet_lag(b, e, nb_nodes):
    inputs = CLRSData()
    inputs['pos'] = ((torch.arange(nb_nodes) * 1.0)/nb_nodes).unsqueeze(0)
    
    inputs['b'] = b.float().unsqueeze(0) 
    inputs['e'] = e.float().unsqueeze(0) 
    inputs['n'] = torch.tensor([nb_nodes]).float()

    s = torch.tensor([])
    t = torch.tensor([])
    
    b = torch.concat((torch.tensor([0]), b))
    e = torch.concat((torch.tensor([0]), e))
    length = 0

    impossible = False

    i = j = nb_nodes
    while i > 0:
        if j == 0:
            impossible = True
            break

        interval = b[j] - (e[i] - b[j]+1)//2
        if e[j - 1] > interval:
            j -= 1
            continue
        
        length += 1
        if e[j - 1] >= interval - 1:
            s = torch.concat((s, e[j-1].unsqueeze(0)))
            t = torch.concat((t, (b[j] - (e[j - 1] == interval - 1 and e[i] - b[j] == 1).long().unsqueeze(0))))
        else:
            s = torch.concat((s, interval.unsqueeze(0)))
            t = torch.concat((t, b[j].unsqueeze(0)))
            s = torch.concat((s, e[j-1].unsqueeze(0)))
            t = torch.concat((t, ((e[j-1]+interval)/2).unsqueeze(0)))
        
        j -= 1
        i = j

    outputs = CLRSData()
    outputs['impossible'] = torch.tensor([impossible]).float()
  
    return CLRSData(inputs=inputs, hints=CLRSData(), length=torch.tensor(length).float(), outputs=outputs, algorithm="jet_lag")

if __name__ == "__main__":
    os.mkdir("tmp/CLRS30/jet_lag")
    
    os.mkdir("tmp/CLRS30/jet_lag/train")

    # Sampling Training set
    nb_nodes_train = torch.randint(16, 16 + 1, (1000,)).tolist()

    train_datapoints = []
    max_length = -1
    for nb_nodes in nb_nodes_train:
        max_activity_dur, max_interval = torch.randint(5, 100, (2,)).tolist()
        b, e = sample_tasks(nb_nodes, max_activity_dur, max_interval)
        data_point = jet_lag(b, e, nb_nodes)
        train_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/jet_lag/val")
    val_datapoints = []
    # Sampling Validation set
    nb_nodes_val = torch.randint(16, 16 + 1, (1000,)).tolist()

    for nb_nodes in nb_nodes_val:
        max_activity_dur, max_interval = torch.randint(5, 100, (2,)).tolist()
        b, e = sample_tasks(nb_nodes, max_activity_dur, max_interval)
        data_point = jet_lag(b, e, nb_nodes)
        val_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length


    os.mkdir("tmp/CLRS30/jet_lag/test")
    test_datapoints = []
    # Sampling Test set
    nb_nodes_test = torch.randint(64, 64 + 1, (1000,)).tolist()

    for nb_nodes in nb_nodes_test:
        max_activity_dur, max_interval = torch.randint(5, 100, (2,)).tolist()
        b, e = sample_tasks(nb_nodes, max_activity_dur, max_interval)
        data_point = jet_lag(b, e, nb_nodes)
        test_datapoints.append(data_point)
        curr_length = data_point.length.item()
        max_length = curr_length if curr_length > max_length else max_length

    for i, data_point in enumerate(train_datapoints):
        torch.save(data_point, f"tmp/CLRS30/jet_lag/train/{i}")

    for i, data_point in enumerate(val_datapoints):
        torch.save(data_point, f"tmp/CLRS30/jet_lag/val/{i}")

    for i, data_point in enumerate(test_datapoints):
        torch.save(data_point, f"tmp/CLRS30/jet_lag/test/{i}")