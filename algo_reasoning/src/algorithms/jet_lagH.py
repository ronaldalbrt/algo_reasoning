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
    
    b = torch.concat((torch.tensor([0]), b))
    e = torch.concat((torch.tensor([0]), e))
    length = 0

    impossible = False

    # i = j = nb_nodes
    # while i > 0:
    #     if j == 0:
    #         impossible = True
    #         break

    #     interval = b[j] - (e[i] - b[j]+1)//2
    #     if e[j - 1] > interval:
    #         j -= 1
    #         continue
        
    #     length += 1
    #     if e[j - 1] >= interval - 1:
    #         s = torch.concat((s, e[j-1].unsqueeze(0)))
    #         t = torch.concat((t, (b[j] - (e[j - 1] == interval - 1 and e[i] - b[j] == 1).long().unsqueeze(0))))
    #     else:
    #         s = torch.concat((s, interval.unsqueeze(0)))
    #         t = torch.concat((t, b[j].unsqueeze(0)))
    #         s = torch.concat((s, e[j-1].unsqueeze(0)))
    #         t = torch.concat((t, ((e[j-1]+interval)/2).unsqueeze(0)))
        
    #     j -= 1
    #     i = j

    for i in range(nb_nodes):
        s = torch.concat((s, e[i].unsqueeze(0)))
        t = torch.concat((t, b[i+1].unsqueeze(0)))

        if i > 0:
            if (s[i] + s[i - 1])/2 < t[i]:
                t[i] = (s[i] + s[i - 1])/2

    print(s)
    print(t)
    outputs = CLRSData()
    outputs['impossible'] = torch.tensor([impossible]).float()
  
    return CLRSData(inputs=inputs, hints=CLRSData(), length=torch.tensor(length).float(), outputs=outputs, algorithm="jet_lag")

if __name__ == "__main__":
    nb_nodes = 3
    b = torch.tensor([30, 60, 120])
    e = torch.tensor([45, 90, 180])

    print(jet_lag(b, e, nb_nodes))