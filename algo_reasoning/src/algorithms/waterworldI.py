import torch
import torch.linalg as LA

from algo_reasoning.src.data import AlgorithmicData
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

def waterworld(n, m, area_percentages, nb_nodes, *args, **kwargs):
    data = AlgorithmicData(algorithm='waterworld', *args, **kwargs)
    
    data.set_inputs({
        'n': torch.tensor(n),
        'm': torch.tensor(m),
        'area_percentages': area_percentages
    }, nb_nodes)

    area_sums = 0
    total_area = n*m

    data.increase_hints({
        'total_area': torch.tensor(total_area),
        'area_sums': torch.tensor(area_sums)
    })
    for i in range(n*m):
        area_sums += area_percentages[i].item()

        data.increase_hints({
            'total_area': torch.tensor(total_area),
            'area_sums': torch.tensor(area_sums)
        })
        
    surface_percentage = area_sums/total_area

    data.set_outputs({
        'surface_percentage': torch.tensor(surface_percentage)
    })

    return data