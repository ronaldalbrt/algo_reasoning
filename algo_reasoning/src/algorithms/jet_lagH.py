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

def jet_lag(b, e, nb_nodes, *args, **kwargs):
    data = CLRSData(algorithm="jet_lag", *args, **kwargs)

    data.set_inputs({
        'b':  b,
        'e':  e,
        'n': torch.tensor(nb_nodes)
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
        'impossible': torch.tensor(impossible)
    })
  
    return data