import torch

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

def jet_lag(b, e, nb_nodes, *args, **kwargs):
    data = CLRSData(algorithm="jet_lag", *args, **kwargs)

    data.set_inputs({
        'b':  b.float().unsqueeze(0),
        'e':  e.float().unsqueeze(0),
        'n': torch.tensor([nb_nodes]).float()
    }, nb_nodes)

    s = torch.tensor([])
    t = torch.tensor([])
    
    b = torch.concat((torch.tensor([0]), b))
    e = torch.concat((torch.tensor([0]), e))

    impossible = False

    i = j = nb_nodes
    while i > 0:
        data.increase_hints({})
        if j == 0:
            impossible = True
            break

        interval = b[j] - (e[i] - b[j]+1)//2
        if e[j - 1] > interval:
            j -= 1
            continue

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

    data.set_outputs({
        'impossible': torch.tensor([impossible]).float()
    })
  
    return data