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
    'N': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
    'impossible': (Stage.OUTPUT, Location.GRAPH, Type.MASK),
    }

def jet_lag(b, e, nb_nodes):
    inputs = CLRSData()
    inputs['pos'] = ((torch.arange(nb_nodes) * 1.0)/nb_nodes).unsqueeze(0)
    
    inputs['b'] = b.float().unsqueeze(0) 
    inputs['e'] = e.float().unsqueeze(0) 
    inputs['N'] = torch.tensor([nb_nodes]).float()

    s = torch.tensor([])
    t = torch.tensor([])

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
        elif e[j - 1] >= interval - 1:
            s = torch.concat((s, e[j-1].unsqueeze()))
            t = torch.concat((t, (b[j] - (e[j - 1] == interval - 1 and e[i] - b[j] == 1).long().unsqueeze())))
        else:
            s = torch.concat((s, interval.unsqueeze()))
            t = torch.concat((t, b[j].unsqueeze()))
            s = torch.concat((s, e[j-1].unsqueeze()))
            t = torch.concat((t, ((e[j-1]+interval)/2).unsqueeze()))
        
        j -= 1
        i = j
  
    return s, t, impossible

if __name__ == "__main__":
    nb_nodes = 3
    b = torch.tensor([30, 60, 120])
    e = torch.tensor([45, 90, 180])

    print(jet_lag(b, e, nb_nodes))